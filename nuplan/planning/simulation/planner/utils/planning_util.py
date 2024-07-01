import math
import numpy as np

def match_projection_points(xy_list: list, frenet_path_node_list: list):

    input_xy_length = len(xy_list)
    frenet_path_length = len(frenet_path_node_list)

    match_point_index_list = np.zeros(input_xy_length, dtype="int32")
    project_node_list = []

    for index_xy in range(input_xy_length):
        x, y = xy_list[index_xy][0], xy_list[index_xy][1]
        start_index = 0
        increase_count = 0
        min_distance = float("inf")

        for i in range(start_index, frenet_path_length):
            frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
            distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
            if distance < min_distance:
                min_distance = distance  # 保留最小值
                match_point_index_list[index_xy] = i
                increase_count = 0
            else:
                increase_count += 1
                if increase_count >= 50:  
                    break
        # 通过匹配点确定投影点
        match_point_index = match_point_index_list[0]
        x_m, y_m, theta_m, k_m = frenet_path_node_list[match_point_index]
        d_v = np.array([x - x_m, y - y_m])
        tau_v = np.array([np.cos(theta_m), np.sin(theta_m)])
        ds = np.dot(d_v, tau_v)
        r_m_v = np.array([x_m, y_m])
        # 根据公式计算投影点的位置信息
        x_r, y_r = r_m_v + ds * tau_v 
        theta_r = theta_m + k_m * ds  
        k_r = k_m  
        project_node_list.append((x_r, y_r, theta_r, k_r))

    return list(match_point_index_list), project_node_list

def cal_projection_s_fun(local_path_opt: list, match_index_list: list, xy_list: list, s_map: list):
    """
    计算若干点的投影对应的s
    """
    projection_s_list = []
    for i in range(len(match_index_list)):
        x, y, theta, kappa = local_path_opt[match_index_list[i]]
        d_v = np.array([xy_list[i][0] - x, xy_list[i][1] - y])
        tou_v = np.array([math.cos(theta), math.sin(theta)]) 
        projection_s_list.append(s_map[match_index_list[i]] + np.dot(d_v, tou_v))  # np.dot(d_v, tou_v)即ds大小,有正负号的

    return projection_s_list

def cal_s_l_fun(obs_xy_list: list, local_path_opt: list, s_map: list):
    """
    这是坐标转换的通用模块
    """
    match_index_list, projection_list = match_projection_points(obs_xy_list, local_path_opt)

    s_list = cal_projection_s_fun(local_path_opt, match_index_list, obs_xy_list, s_map)

    l_list = []

    for i in range(len(obs_xy_list)):
        pro_x, pro_y, theta, kappa = projection_list[i]  # 投影点的信息
        n_r = np.array([-math.sin(theta), math.cos(theta)])  # 投影点的单位法向量
        x, y = obs_xy_list[i][0], obs_xy_list[i][1]  # 待投影的位置
        r_h = np.array([x, y])  # 车辆实际位置的位矢
        r_r = np.array([pro_x, pro_y])  # 投影点的位矢
        # 核心公式: l*n_r = r_h - r_r
        l_list.append(np.dot(r_h - r_r, n_r))  # *UE4定义的是左手系，所以在车辆左侧的为负值(SL图是反的)*

    return s_list, l_list

def cal_dy_obs_derivative(l_set, vx_set, vy_set, proj_heading_set, proj_kappa_set):
    """
        该函数将计算frenet坐标系下动态障碍物的s_dot, l_dot, dl/ds
    """
    n = 128
    # 输出初始化
    s_dot_set = np.ones(n) * np.nan
    l_dot_set = np.ones(n) * np.nan
    dl_set = np.ones(n) * np.nan

    for i in range(len(l_set)):
        if np.isnan(l_set[i]):
            break
        v_h = np.array([vx_set[i], vy_set[i]])
        n_r = np.array([-np.sin(proj_heading_set[i]), np.cos(proj_heading_set[i])])
        t_r = np.array([np.cos(proj_heading_set[i]), np.sin(proj_heading_set[i])])
        l_dot_set[i] = np.dot(v_h, n_r)
        s_dot_set[i] = np.dot(v_h, t_r) / (1 - proj_kappa_set[i] * l_set[i])
        if abs(s_dot_set[i]) < 1e-6:
            dl_set[i] = 0
        else:
            dl_set[i] = l_dot_set[i] / s_dot_set[i]

    return s_dot_set, l_dot_set, dl_set

def find_match_points(xy_list: list, frenet_path_node_list: list, is_first_run: bool, pre_match_index: int):
    """
    计算笛卡尔坐标系（直角坐标系）下的位置(x,y)，在frenet（自然坐标系）路径上的匹配点索引和投影点位置信息
    即输入若干个坐标，返回他们的匹配点索引值和在frenet曲线上的投影点
    return:
            match_point_index_list 匹配点在frenet_path下的编号的集合(即从全局路径第一个点开始数，第几个点是匹配点)
            project_node_list 投影点的信息， project_node_list = [(x_p0, y_p0, heading_p0, kappa_p0), ...]
    """

    i = -1
    input_xy_length = len(xy_list)
    frenet_path_length = len(frenet_path_node_list)

    match_point_index_list = np.zeros(input_xy_length, dtype="int32")
    project_node_list = []

    if is_first_run is True:
        for index_xy in range(input_xy_length):  # 为每一个点寻找匹配点
            x, y = xy_list[index_xy]
            start_index = 0
            # 用increase_count记录distance连续增大的次数，避免多个局部最小值的干扰
            increase_count = 0
            min_distance = float("inf")
            # 确定匹配点
            for i in range(start_index, frenet_path_length):
                frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
                # 计算(x,y)与(frenet_node_x, frenet_node_y)之间的距离
                distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
                if distance < min_distance:
                    min_distance = distance  # 保留最小值
                    match_point_index_list[index_xy] = i
                    increase_count = 0
                else:
                    increase_count += 1
                    if increase_count >= 50:  # 向后50个点还没找到的话，说明当前最小值点及时最优的
                        # 第一次运行阈值较大，是为了保证起始点匹配的精确性
                        break
            # 通过匹配点确定投影点
            """
            (x, y)'表示向量转置
            d_v是匹配点（x_m, y_m）指向待投影点（x,y）的向量（x-x_m, y-y_m）
            tou_v是匹配点的单位切向量(cos(theta_m), sin(theta_m))'
            (x_r, y_r)' 约等于 (x_m, y_m)' + (d_v . tou_v)*tou_v
            k_r 约等于 k_m， 投影点曲率
            theta_r 约等于 theta_m + k_m*(d_v . tou_v)， 投影点切线与坐标轴夹角
            具体原理看老王《自动驾驶决策规划算法》第一章第三节
            """
            match_point_index = match_point_index_list[0]
            x_m, y_m, theta_m, k_m = frenet_path_node_list[match_point_index]
            d_v = np.array([x - x_m, y - y_m])
            tou_v = np.array([np.cos(theta_m), np.sin(theta_m)])
            ds = np.dot(d_v, tou_v)
            r_m_v = np.array([x_m, y_m])
            # 根据公式计算投影点的位置信息
            x_r, y_r = r_m_v + ds * tou_v  # 计算投影点坐标
            theta_r = theta_m + k_m * ds  # 计算投影点在frenet曲线上切线与X轴的夹角
            k_r = k_m  # 投影点在frenet曲线处的曲率
            # 将结果打包放入缓存区
            project_node_list.append((x_r, y_r, theta_r, k_r))

    else:
        for index_xy in range(input_xy_length):
            x, y = xy_list[index_xy]
            start_index = pre_match_index

            # 用increase_count记录distance连续增大的次数，避免多个局部最小值的干扰
            # TODO 判断是否有上个周期的index结果，没有的话start_index=1, increase_count_limit=50
            increase_count_limit = 5
            increase_count = 0
            # 上个周期匹配点坐标
            pre_match_point_xy = [frenet_path_node_list[start_index][0], frenet_path_node_list[start_index][1]]
            pre_match_point_theta_m = frenet_path_node_list[start_index][2]
            # 上个匹配点在曲线上的切向向量
            pre_match_point_direction = np.array([np.cos(pre_match_point_theta_m), np.sin(pre_match_point_theta_m)])
            # 计算上个匹配点指向当前(x, y）的向量
            pre_match_to_xy_v = np.array([x - pre_match_point_xy[0], y - pre_match_point_xy[1]])
            # 计算pre_match_to_xy_v在pre_match_point_direction上的投影，用于判断遍历方向
            flag = np.dot(pre_match_to_xy_v, pre_match_point_direction)

            min_distance = float("inf")
            if flag > 0:  # 正向遍历
                for i in range(start_index, frenet_path_length):
                    frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
                    # 计算（x,y) 与 （frenet_node_x, frenet_node_y） 之间的距离
                    distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
                    if distance < min_distance:
                        min_distance = distance  # 保留最小值
                        match_point_index_list[index_xy] = i
                        increase_count = 0
                    else:
                        increase_count += 1
                        if increase_count >= increase_count_limit:  # 为了加快速度，这里阈值为5，第一个周期是不同的，向后5个点还没找到的话，说明当前最小值点及时最优的
                            break
            else:  # 反向遍历
                for i in range(start_index, -1, -1):  # range(start,end,step)，其中start为闭，end为开，-1为步长
                    frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
                    # 计算（x,y) 与 （frenet_node_x, frenet_node_y） 之间的距离
                    distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
                    if distance < min_distance:
                        min_distance = distance  # 保留最小值
                        match_point_index_list[index_xy] = i
                        increase_count = 0
                    else:
                        increase_count += 1
                        if increase_count >= increase_count_limit:
                            break
            # 通过匹配点确定投影点
            match_point_index = match_point_index_list[0]
            x_m, y_m, theta_m, k_m = frenet_path_node_list[match_point_index]
            d_v = np.array([x - x_m, y - y_m])
            tou_v = np.array([np.cos(theta_m), np.sin(theta_m)])
            ds = np.dot(d_v, tou_v)
            r_m_v = np.array([x_m, y_m])
            # 根据公式计算投影点的坐标信息
            x_r, y_r = r_m_v + ds * tou_v
            theta_r = theta_m + k_m * ds
            k_r = k_m
            # 将结果打包放入缓存区
            project_node_list.append((x_r, y_r, theta_r, k_r))

    return list(match_point_index_list), project_node_list