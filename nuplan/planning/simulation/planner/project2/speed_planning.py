import numpy as np
import math
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint


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

    def GenerateSTGraph(self, obs_set: []):
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