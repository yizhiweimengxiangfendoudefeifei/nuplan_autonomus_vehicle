import math
import logging
from typing import List, Type, Optional, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.planning.simulation.planner.project2.simple_predictor import SimplePredictor
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor

from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects

from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class MyPlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
    ):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """

        # 1. Routing
        ego_state, observations = current_input.history.current_state
        if not self._routing_complete:
            self._router._initialize_ego_path(ego_state, self.max_velocity)
            self._routing_complete = True

        # 2. Generate reference line
        self._reference_path_provider = ReferenceLineProvider(self._router)
        self._reference_path_provider._reference_line_generate(ego_state)

        # 3. Objects prediction
        self._predictor = SimplePredictor(ego_state, observations, self.horizon_time.time_s, self.sampling_time.time_s)
        objects = self._predictor.predict()

        # 4. Planning
        trajectory: List[EgoState] = self.planning(ego_state, self._reference_path_provider, objects,
                                                    self.horizon_time, self.sampling_time, self.max_velocity)

        return InterpolatedTrajectory(trajectory)

    # TODO: 2. Please implement your own trajectory planning.
    def planning(self,
                 ego_state: EgoState,
                 reference_path_provider: ReferenceLineProvider,
                 objects: List[TrackedObjects],
                 horizon_time: TimePoint,
                 sampling_time: TimePoint,
                 max_velocity: float) -> List[EgoState]:
        """
        Implement trajectory planning based on input and output, recommend using lattice planner or piecewise jerk planner.
        param: ego_state Initial state of the ego vehicle
        param: reference_path_provider Information about the reference path
        param: objects Information about dynamic obstacles
        param: horizon_time Total planning time
        param: sampling_time Planning sampling time
        param: max_velocity Planning speed limit (adjustable according to road speed limits during planning process)
        return: trajectory Planning result
        """






        
        # 可以实现基于采样的planer或者横纵向解耦的planner，此处给出planner的示例，仅提供实现思路供参考
        # 1.Path planning
        optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s = self.path_planning( \
            ego_state, reference_path_provider)

        # 2.Transform path planning result to cartesian frame
        path_idx2s, path_x, path_y, path_heading, path_kappa = transform_path_planning(optimal_path_s, optimal_path_l, \
                                                                                       optimal_path_dl,
                                                                                       optimal_path_ddl, \
                                                                                       reference_path_provider)

        # 3.Speed planning
        optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t = speed_planning( \
            ego_state, horizon_time.time_s, max_velocity, objects, \
            path_idx2s, path_x, path_y, path_heading, path_kappa)

        # 4.Produce ego trajectory
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            ),
            tire_steering_angle=ego_state.dynamic_car_state.tire_steering_rate,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for iter in range(int(horizon_time.time_us / sampling_time.time_us)):
            relative_time = (iter + 1) * sampling_time.time_s
            # 根据relative_time 和 speed planning 计算 velocity accelerate （三次多项式）
            s, velocity, accelerate = cal_dynamic_state(relative_time, optimal_speed_t, optimal_speed_s,
                                                        optimal_speed_s_dot, optimal_speed_s_2dot)
            # 根据当前时间下的s 和 路径规划结果 计算 x y heading kappa （线形插值）
            x, y, heading, _ = cal_pose(s, path_idx2s, path_x, path_y, path_heading, path_kappa)

            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(velocity, 0),
                rear_axle_acceleration_2d=StateVector2D(accelerate, 0),
                tire_steering_angle=heading,
                time_point=state.time_point + sampling_time,
                vehicle_parameters=state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=0,
                angular_accel=0,
            )

            trajectory.append(state)
        







        trajectory = []
        return trajectory
    
    def path_planning(self, 
                      ego_state: EgoState, 
                      reference_path_provider: ReferenceLineProvider):
        """
        Path planning based on the reference path.
        :param ego_state: Ego state.
        :param reference_path_provider: Reference path provider.
        :return: l, dl, ddl, s
        """
        
        optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s = \
                self.PathQuadraticProgramming(ego_state, reference_path_provider)
        return optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s
    
    def PathQuadraticProgramming(self, 
                                 ego_state: EgoState, 
                                 reference_path_provider: ReferenceLineProvider):
        """
        Path planning based on the reference path using quadratic programming.
        :param ego_state: Ego state.
        :param reference_path_provider: Reference path provider.
        """
        n = len(reference_path_provider._x_of_reference_line)
        s = 0.0
        ds = 0.2
        optimal_path_l = []
        optimal_path_dl = []
        optimal_path_ddl = []
        optimal_path_s = []
        print(reference_path_provider._s_of_reference_line[-1])
        while (s <= reference_path_provider._s_of_reference_line[-1]):
            optimal_path_s.append(s)
            optimal_path_l.append(0.0)
            optimal_path_dl.append(0.0)
            optimal_path_ddl.append(0.0)
            s += ds
        return optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s
        # 输出初始化
        '''
        n = len(reference_path_provider._x_of_reference_line)
        if len(reference_path_provider._s_of_reference_line) < 2:
            print("Reference path is too short.")
            return None, None, None, None
        w = 1.0 # 车辆宽度
        ds = reference_path_provider._s_of_reference_line[1] - reference_path_provider._s_of_reference_line[0]
        qp_path_l = np.zeros(n)
        qp_path_dl = np.zeros(n)
        qp_path_ddl = np.zeros(n)
        qp_path_s = np.zeros(n)

        # Hissen矩阵初始化
        H = csr_matrix((3 * n, 3 * n))
        H_L = csr_matrix((3 * n, 3 * n))
        H_DL = csr_matrix((3 * n, 3 * n))
        H_DDL = csr_matrix((3 * n, 3 * n))
        H_DDDL = csr_matrix((n - 1, 3 * n))
        H_CENTRE = csr_matrix((3 * n, 3 * n))
        H_L_END = csr_matrix((3 * n, 3 * n))
        H_DL_END = csr_matrix((3 * n, 3 * n))
        H_DDL_END = csr_matrix((3 * n, 3 * n))

        # Gradient matrix f
        f = np.zeros(3 * n)

        # 等式约束Aeq*x = beq,连续性约束
        Aeq = csr_matrix((2 * n - 2, 3 * n))
        beq = np.zeros(2 * n - 2)

        # 动力学约束
        lb = np.ones(3 * n) * (-float("inf"))
        ub = np.ones(3 * n) * float("inf")

        # 合并约束矩阵
        A_merge = csr_matrix((2 * n - 2 + 3 * n, 3 * n))
        lb_merge = np.zeros(2 * n - 2 + 3 * n)
        ub_merge = np.zeros(2 * n - 2 + 3 * n)

        # 生成 H_L, H_DL, H_DDL, H_CENTRE
        for i in range(n):
            H_L[3 * i, 3 * i] = 1
            H_DL[3 * i + 1, 3 * i + 1] = 1
            H_DDL[3 * i + 2, 3 * i + 2] = 1

        H_CENTRE = H_L

        # 生成 H_DDDL
        for i in range(n - 1):
            row = i
            col = 3 * i
            H_DDDL[row, col + 2] = 1
            H_DDDL[row, col + 5] = -1

        # 生成 H_L_END, H_DL_END, H_DDL_END
        H_L_END[3 * n - 3, 3 * n - 3] = 0
        H_DL_END[3 * n - 2, 3 * n - 2] = 0
        H_DDL_END[3 * n - 1, 3 * n - 1] = 0

        H = Config_.qp_cost_l * (H_L.transpose() * H_L) + \
            Config_.qp_cost_dl * (H_DL.transpose() * H_DL) + \
            Config_.qp_cost_ddl * (H_DDL.transpose() * H_DDL) + \
            Config_.qp_cost_dddl * (H_DDDL.transpose() * H_DDDL) / (ds**2) + \
            Config_.qp_cost_centre * (H_CENTRE.transpose() * H_CENTRE) + \
            Config_.qp_cost_end_l * (H_L_END.transpose() * H_L_END) + \
            Config_.qp_cost_end_dl * (H_DL_END.transpose() * H_DL_END) + \
            Config_.qp_cost_end_ddl * (H_DDL_END.transpose() * H_DDL_END)
        H = 2 * H

        #  生成 f
        center_line = reference_path_provider._discrete_path
        for i in range(n):
            f[3 * i] = -2.0 * 0.0
            # 合理调节centerline的权重
            if (math.fabs(f[3 * i]) > 0.3):
                f[3 * i] = Config_.qp_cost_centre
        
        # 不等式约束
        row_index_start = 0
        for i in range(n - 1):
            row = row_index_start + 2 * i
            col = 3 * i
            A_merge[row, col] = 1
            A_merge[row, col + 1] = ds
            A_merge[row, col + 2] = ds**2 / 3
            A_merge[row, col + 3] = -1
            A_merge[row, col + 5] = ds**2 / 6

            A_merge[row + 1, col + 1] = 1
            A_merge[row + 1, col + 2] = ds / 2
            A_merge[row + 1, col + 4] = -1
            A_merge[row + 1, col + 5] = ds / 2

        row_index_start = 2 * n - 2

        for i in range(n):
            row = row_index_start + 3 * i
            col = 3 * i
            A_merge[row, col] = 1
            A_merge[row + 1, col + 1] = 1
            A_merge[row + 2, col + 2] = 1

        # 生成 lb ub 主要是对规划起点做约束
        lb = np.zeros((3 * n, 1))
        ub = np.zeros((3 * n, 1))

        lb[0] = 0.0
        lb[1] = 0.0
        lb[2] = 0.0
        ub[0] = lb[0]
        ub[1] = lb[1]
        ub[2] = lb[2]

        for i in range(1, n):
            lb[3 * i] = 0.0 + w / 2
            lb[3 * i + 1] = -2
            lb[3 * i + 2] = -0.1
            ub[3 * i] = 0.0 - w / 2
            ub[3 * i + 1] = 2
            ub[3 * i + 2] = 0.1

        # 求解
        init_variables = np.zeros((3 * n,))
        '''
