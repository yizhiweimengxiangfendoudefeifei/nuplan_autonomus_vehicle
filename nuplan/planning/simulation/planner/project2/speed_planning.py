import numpy as np
import math
from typing import List
from scipy.interpolate import interp1d
import cvxopt
import cvxopt.solvers

from nuplan.common.actor_state.state_representation import TimePoint


class FrenetState:
    def __init__(self):
        self.s = 0.0
        self.s_d = 0.0
        self.s_dd = 0.0
        self.l = 0.0
        self.l_d = 0.0
        self.l_dd = 0.0

class SpeedPlanning:
    def __init__(
            self, 
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,):
        """
        Constructor for the SpeedPlanning class.

        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity

    def generate_st_graph(self, obs_set: List[FrenetState]):
        """
        Generate the ST graph for the speed planning.
        """
        n = len(obs_set)
        obs_st_s_in_set = np.ones(n) * np.nan
        obs_st_s_out_set = np.ones(n) * np.nan
        obs_st_t_in_set = np.ones(n) * np.nan
        obs_st_t_out_set = np.ones(n) * np.nan
        for i in range(n):
            if math.abs(obs_set[i].l_d < 0.3):
                continue
            t_zero = -obs_set[i].l / obs_set[i].l_d
            t_boundary1 = 2 / obs_set[i].l_d + t_zero
            t_boundary2 = -2 / obs_set[i].l_d + t_zero
            if t_boundary1 > t_boundary2:
                t_max = t_boundary1
                t_min = t_boundary2
            else:
                t_max = t_boundary2
                t_min = t_boundary1
            if t_max < 1 or t_min > 8:
                continue
            if t_min < 0 and t_max > 0:
                obs_st_s_in_set[i] = obs_set[i].s
                # 匀速运动
                obs_st_s_out_set[i] = obs_set[i].s + obs_set[i].s_d * t_max
                obs_st_t_in_set[i] = 0
                obs_st_t_out_set[i] = t_max
            else:
                # 正常障碍物
                obs_st_s_in_set[i] = obs_set[i].s + obs_set[i].s_d * t_min
                obs_st_s_out_set[i] = obs_set[i].s + obs_set[i].s_d * t_max
                obs_st_t_in_set[i] = t_min
                obs_st_t_out_set[i] = t_max

        return obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set
    
    def speed_DP(self, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, plan_start_s_dot,
                 reference_speed=50, w_cost_ref_speed=4000, w_cost_accel=100, w_cost_obs=1e7):
        """
        Dynamic programming for speed planning.
        """
        s_list = np.concatenate((np.arange(0, 5, 0.5), np.arange(5.5, 15, 1), np.arange(16, 30, 1.5), np.arange(32, 55, 2.5)))
        t_list = np.arange(0.5, 8.5, 0.5)
        dp_st_cost = np.ones((40, 16)) * np.inf
        dp_st_s_dot = np.zeros((40, 16))
        dp_st_node = np.zeros((40, 16))

        # 计算从起点到第一列的代价
        for i in range(len(s_list)):
            dp_st_cost[i, 0] = self.calc_dp_cost(0, 0, i, 0, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                                        obs_st_t_out_set, w_cost_ref_speed, reference_speed,
                                        w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot)
            s_end, t_end = CalcSTCoordinate(i, 0, s_list, t_list)
            dp_st_s_dot[i, 0] = s_end / t_end
        
        # 动态规划主程序
        for i in range(1, len(t_list)):
            for j in range(len(s_list)):
                cur_row = j
                cur_col = i
                for k in range(len(s_list)):
                    pre_row = k
                    pre_col = i - 1
                    cost_temp = self.calc_dp_cost(pre_row, pre_col, cur_row, cur_col, obs_st_s_in_set, obs_st_s_out_set,
                                        obs_st_t_in_set, obs_st_t_out_set, w_cost_ref_speed, reference_speed,
                                        w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot)
                    if cost_temp + dp_st_cost[pre_row, pre_col] < dp_st_cost[cur_row, cur_col]:
                        dp_st_cost[cur_row, cur_col] = cost_temp + dp_st_cost[pre_row, pre_col]
                        s_start, t_start = CalcSTCoordinate(pre_row, pre_col, s_list, t_list)
                        s_end, t_end = CalcSTCoordinate(cur_row, cur_col, s_list, t_list)
                        dp_st_s_dot[cur_row, cur_col] = (s_end - s_start) / (t_end - t_start)
                        dp_st_node[cur_row, cur_col] = pre_row
        # 输出初始化
        dp_speed_s = np.ones(len(t_list)) * np.nan
        dp_speed_t = dp_speed_s
        min_cost = np.inf
        min_row = np.inf
        min_col = np.inf

        for i in range(len(s_list)):
            if dp_st_cost[i, len(t_list) - 1] <= min_cost:
                min_cost = dp_st_cost[i, len(t_list) - 1]
                min_row = i
                min_col = len(t_list) - 1

        for j in range(len(t_list)):
            if dp_st_cost[0, j] <= min_cost:
                min_cost = dp_st_cost[0, j]
                min_row = 0
                min_col = j
        dp_speed_s[min_col], dp_speed_t[min_col] = CalcSTCoordinate(min_row,min_col,s_list,t_list)

        # 反向回溯
        while min_col != 0:
            pre_row = dp_st_node[min_row, min_col]
            pre_col = min_col - 1
            dp_speed_s[pre_col], dp_speed_t[pre_col] = CalcSTCoordinate(pre_row,pre_col,s_list,t_list)
            min_row = pre_row
            min_col = pre_col

        return dp_speed_s, dp_speed_t
        
    def calc_dp_cost(self, row_start, col_start, row_end, col_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
               obs_st_t_out_set, w_cost_ref_speed, reference_speed, w_cost_accel, w_cost_obs, plan_start_s_dot,
               s_list, t_list, dp_st_s_dot):

        # 首先计算终点的st坐标
        s_end, t_end = CalcSTCoordinate(row_end, col_end, s_list, t_list)
        # 规定起点的行列号为0
        if row_start == 0:
            # 边的起点为dp的起点
            s_start = 0
            t_start = 0
            s_dot_start = plan_start_s_dot
        else:
            # 边的起点不是dp的起点
            s_start, t_start = CalcSTCoordinate(row_start, col_start, s_list, t_list)
            s_dot_start = dp_st_s_dot[row_start][col_start]
        cur_s_dot = (s_end - s_start) / (t_end - t_start)
        cur_s_dot2 = (cur_s_dot - s_dot_start) / (t_end - t_start)
        # 计算推荐速度代价
        cost_ref_speed = w_cost_ref_speed * (cur_s_dot - reference_speed) ** 2
        # 计算加速度代价，这里注意，加速度不能超过车辆动力学上下限
        if 4 > cur_s_dot2 > -6:
            cost_accel = w_cost_accel * cur_s_dot2 ** 2
        else:
            # 超过车辆动力学限制，代价会增大很多倍
            cost_accel = 100000 * w_cost_accel * cur_s_dot2 ** 2
        cost_obs = self.calc_obs_cost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set,
                            obs_st_t_in_set, obs_st_t_out_set, w_cost_obs)
        cost = cost_obs + cost_accel + cost_ref_speed

        return cost
    
    def calc_obs_cost(self, s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, w_cost_obs):
        obs_cost = 0
        n = 5
        dt = (t_end - t_start)/(n - 1)
        # 边的斜率
        k = (s_end - s_start)/(t_end - t_start)
        for i in range(n):
            # 计算采样点的坐标
            t = t_start + (i - 1) * dt
            s = s_start + k * (i - 1) * dt
            # 遍历所有障碍物
            for j in range(len(obs_st_s_in_set)):
                if np.isnan(obs_st_s_in_set[j]):
                    continue
                # 计算点到st线段的最短距离
                vector1 = np.array([obs_st_s_in_set[j], obs_st_t_in_set[j]]) - np.array([s, t])
                vector2 = np.array([obs_st_s_out_set[j], obs_st_t_out_set[j]]) - np.array([s, t])
                vector3 = vector2 - vector1
                min_dis = 0
                dis1 = np.sqrt(vector1.dot(vector1))
                dis2 = np.sqrt(vector2.dot(vector2))
                dis3 = abs(vector1[0]*vector3[1] - vector1[1]*vector3[0])/np.sqrt(vector3.dot(vector3))
                if (vector1.dot(vector3) > 0 and vector2.dot(vector3) > 0) or (vector1.dot(vector3) < 0 and vector2.dot(vector3) < 0):
                    min_dis = min(dis1, dis2)
                else:
                    min_dis = dis3
                obs_cost = obs_cost + self.calc_collision_cost(w_cost_obs, min_dis)

        return obs_cost
    
    def calc_collision_cost(self, w_cost_obs, min_dis):
        if abs(min_dis) < 0.5:
            collision_cost = w_cost_obs
        elif 0.5 < abs(min_dis) < 1.5:
            # min_dis = 0.5    collision_cost = w_cost_obs**1
            # min_dis = 1.5    collision_cost = w_cost_obs**0 = 1
            collision_cost = w_cost_obs ** ((0.5 - min_dis) + 1)
        else:
            collision_cost = 0

        return collision_cost
    

    def generate_convex_space(self, dp_speed_s, dp_speed_t, path_index2s, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                          obs_st_t_out_set, trajectory_kappa_init, max_lateral_accel=0.2*9.8):
        """
            该函数将计算出s,s_dot的上下界，开辟凸空间供二次规划使用
        """
        n = 16
        s_lb = np.ones(n) * -np.inf
        s_ub = np.ones(n) * np.inf
        s_dot_lb = np.ones(n) * -np.inf
        s_dot_ub = np.ones(n) * np.inf
        path_index2s_end_index = len(path_index2s)
        dp_speed_end_index = len(dp_speed_s)

        for k in range(1, len(path_index2s)):
            if path_index2s[k] == 0 and path_index2s[k - 1] != 0:
                path_index2s_end_index = k - 1
                break
            path_index2s_end_index = k

        for k in range(len(dp_speed_s)):
            if np.isnan(dp_speed_s[k]):
                dp_speed_end_index = k - 1
                break

        for i in range(n):
            if np.isnan(dp_speed_s[i]):
                break
            cur_s = dp_speed_s[i]
            cur_kappa = interp1d(path_index2s[0:path_index2s_end_index], trajectory_kappa_init[0:path_index2s_end_index])(cur_s)
            max_speed = np.sqrt(max_lateral_accel / (abs(cur_kappa) + 1e-10))
            min_speed = 0
            s_dot_lb[i] = min_speed
            s_dot_ub[i] = max_speed

        for i in range(len(obs_st_s_in_set)):
            if np.isnan(obs_st_s_in_set[i]):
                continue
            obs_t = (obs_st_t_in_set[i] + obs_st_t_out_set[i]) / 2
            obs_s = (obs_st_s_in_set[i] + obs_st_s_out_set[i]) / 2
            # 计算障碍物的纵向速度
            obs_speed = (obs_st_s_out_set[i] - obs_st_s_in_set[i]) / (obs_st_t_out_set[i] - obs_st_t_in_set[i])
            dp_s = interp1d([0] + list(dp_speed_t[0:dp_speed_end_index]), [0] + list(dp_speed_s[0:dp_speed_end_index]))(obs_t)

            # 找到dp_speed_t中 与obs_st_t_in_set(i)最近的时间，并将此时间的编号赋值给t_lb_index
            time1 = 0
            for j in range(len(dp_speed_t) - 1):
                if dp_speed_t[0] > obs_st_t_in_set[i]:
                    time1 = j
                    break
                elif dp_speed_t[j] <= obs_st_t_in_set[i] < dp_speed_t[j + 1]:
                    time1 = j
                    break
            t_lb_index = time1

            # 找到dp_speed_t中 与obs_st_t_out_set(i)最近的时间，并将此时间的编号赋值给t_ub_index
            time2 = 0
            for j in range(len(dp_speed_t) - 1):
                if dp_speed_t[0] > obs_st_t_out_set[i]:
                    time2 = j
                    break
                elif dp_speed_t[j] <= obs_st_t_out_set[i] < dp_speed_t[j + 1]:
                    time2 = j
                    break
            t_ub_index = time2

            # 这里稍微做个缓冲，把 t_lb_index 稍微缩小一些，t_ub_index稍微放大一些
            t_lb_index = max(t_lb_index - 2, 3)     # 最低为3 因为碰瓷没法处理
            t_ub_index = min(t_ub_index + 2, dp_speed_end_index)

            if obs_s > dp_s:
                # 减速避让
                for m in range(t_lb_index, t_ub_index + 1):
                    dp_t = dp_speed_t[m]
                    s_ub[m] = min(s_ub[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))
            else:
                # 加速超车
                for m in range(t_lb_index, t_ub_index + 1):
                    dp_t = dp_speed_t[m]
                    s_lb[m] = max(s_lb[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))

        return s_lb, s_ub, s_dot_lb, s_dot_ub
    
    def speed_QP(self, plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t, s_lb, s_ub, s_dot_lb, s_dot_ub,
             w_cost_s_dot2=10, w_cost_v_ref=50, w_cost_jerk=500, reference_speed=50):
        """
            速度二次规划
        """
        dp_speed_end = 16
        for i in range(len(dp_speed_s)):
            if np.isnan(dp_speed_s[i]):
                dp_speed_end = i - 1
                break

        n = 17
        qp_s_init = np.ones(n) * np.nan
        qp_s_dot_init = np.ones(n) * np.nan
        qp_s_dot2_init = np.ones(n) * np.nan
        relative_time_init = np.ones(n) * np.nan

        s_end = dp_speed_s[dp_speed_end]
        recommend_T = dp_speed_t[dp_speed_end]
        qp_size = dp_speed_end + 1

        # 连续性约束矩阵初始化，这里的等式约束应该是 Aeq'*X == beq 有转置
        Aeq = cvxopt.matrix(np.zeros((3 * qp_size, 2 * qp_size - 2)))
        beq = cvxopt.matrix(np.zeros((2 * qp_size - 2, 1)))
        # 不等式约束初始化
        lb = cvxopt.matrix(np.ones(3 * qp_size))
        ub = lb

        # 计算采样间隔时间dt
        dt = recommend_T / dp_speed_end

        A_sub = np.array([[1, 0],
                        [dt, 1],
                        [(1 / 3) * dt ** 2, (1 / 2) * dt],
                        [-1, 0],
                        [0, -1],
                        [(1 / 6) * dt ** 2, dt / 2]])

        for i in range(qp_size - 1):
            Aeq[3 * i:3 * i + 6, 2 * i:2 * i + 2] = A_sub

        # 这里初始化不允许倒车约束，也就是 s(i) - s(i+1) <= 0
        A = np.zeros((qp_size - 1, 3 * qp_size))
        b = np.zeros((qp_size - 1, 1))

        for i in range(qp_size - 1):
            A[i, 3 * i] = 1
            A[i, 3 * i + 3] = -1

        # 由于生成的凸空间约束s_lb s_ub不带起点，所以lb(i) = s_lb(i-1) 以此类推
        # 允许最小加速度为-6 最大加速度为4(基于车辆动力学)
        for i in range(1, qp_size):
            lb[3 * i] = s_lb[i - 1]
            lb[3 * i + 1] = s_dot_lb[i - 1]
            lb[3 * i + 2] = -6
            ub[3 * i] = s_ub[i - 1]
            ub[3 * i + 1] = s_dot_ub[i - 1]
            ub[3 * i + 2] = 4

        # 起点约束
        lb[0] = 0
        lb[1] = plan_start_s_dot
        lb[2] = plan_start_s_dot2
        ub[0] = lb[0]
        ub[1] = lb[1]
        ub[2] = lb[2]

        # 加速度代价 jerk代价 以及推荐速度代价
        A_s_dot2 = np.zeros((3 * qp_size, 3 * qp_size))
        A_jerk = np.zeros((3 * qp_size, qp_size - 1))
        A_ref = np.zeros((3 * qp_size, 3 * qp_size))

        A4_sub = np.array([[0], [0], [1], [0], [0], [-1]])

        for i in range(1, qp_size + 1):
            A_s_dot2[3 * i - 1, 3 * i - 1] = 1
            A_ref[3 * i - 2, 3 * i - 2] = 1

        for i in range(1, qp_size):
            A_jerk[3 * i - 3:3 * i + 3, i - 1:i] = A4_sub

        # 生成H F
        H = w_cost_s_dot2 * (A_s_dot2 @ A_s_dot2.T) + w_cost_jerk * (A_jerk @ A_jerk.T) + w_cost_v_ref * (A_ref @ A_ref.T)
        H = 2 * H
        f = cvxopt.matrix(np.zeros((3 * qp_size, 1)))
        for i in range(1, qp_size + 1):
            f[3 * i - 2] = -2 * w_cost_v_ref * reference_speed

        # 计算二次规划
        sol = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f), cvxopt.matrix(A), cvxopt.matrix(b), Aeq, beq)
        X = np.array(sol['x'])

        for i in range(1, qp_size + 1):
            qp_s_init[i - 1] = X[3 * i - 3]
            qp_s_dot_init[i - 1] = X[3 * i - 2]
            qp_s_dot2_init[i - 1] = X[3 * i - 1]
            relative_time_init[i - 1] = (i - 1) * dt

        return qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init
    
    def increase_points(self, s_init, s_dot_init, s_dot2_init, relative_time_init):
        """
        该函数将增密s s_dot s_dot2
        """

        # Find t_end
        t_end = len(relative_time_init)
        for i in range(len(relative_time_init)):
            if np.isnan(relative_time_init[i]):
                t_end = i - 1
                break
        T = relative_time_init[t_end]

        # Densify
        n = 401
        dt = T / (n - 1)
        s = np.zeros(n)
        s_dot = np.zeros(n)
        s_dot2 = np.zeros(n)
        relative_time = np.zeros(n)

        tmp = 0
        for i in range(n):
            current_t = (i - 1) * dt
            for j in range(t_end - 1):
                if relative_time_init[j] <= current_t < relative_time_init[j + 1]:
                    tmp = j
                    break
            x = current_t - relative_time_init[tmp]
            s[i] = s_init[tmp] + s_dot_init[tmp] * x + (1 / 3) * s_dot2_init[tmp] * x ** 2 + (1 / 6) * s_dot2_init[tmp + 1] * x ** 2
            s_dot[i] = s_dot_init[tmp] + 0.5 * s_dot2_init[tmp] * x + 0.5 * s_dot2_init[tmp + 1] * x
            s_dot2[i] = s_dot2_init[tmp] + (s_dot2_init[tmp + 1] - s_dot2_init[tmp]) * x / (
                        relative_time_init[tmp + 1] - relative_time_init[tmp])
            relative_time[i] = current_t

        return s, s_dot, s_dot2, relative_time
