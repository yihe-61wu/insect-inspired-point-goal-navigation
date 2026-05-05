from typing import TYPE_CHECKING, Dict, Union, cast

from math import pi
import numpy as np
from numpy import bool_, int64, ndarray
import copy

import habitat
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

from habitat.core.simulator import Observations
from habitat.core.embodied_task import Metrics
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from utils import normalize_angle, queue
from neuro_utils import MultiOutputMushroomBody


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, config):
        # config should be the object returned by habitat.get_config()
        self.name = self.__class__.__name__
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=config.habitat.task.measurements.success.success_distance,
            return_one_hot=False,
        )
        self.no_more_current_episode = True

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def memory_consolidation(self, info: Metrics):
        r"""This method is not used in this agent."""
        pass
        
    def reset(self) -> None:
        pass


class RandomAgent(Agent):
    def __init__(self, config):
        # config should be the object returned by habitat.get_config()
        self.name = self.__class__.__name__
        self.goal_radius=config.habitat.task.measurements.success.success_distance
        self.action_history = queue(9, HabitatSimActions.move_forward)
        self.reset()

    @property
    def no_more_current_episode(self):
        return False

    def reset(self) -> None:
        self.action_history.reset()

    def is_goal_reached(self, observations: Observations) -> bool_:
        dist = observations['pointgoal_with_gps_compass'][0]
        return dist <= self.goal_radius

    def act(self, observations: Observations) -> Dict[str, int64]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            action = np.random.choice(
                [
                    HabitatSimActions.move_forward,
                    HabitatSimActions.turn_left,
                    HabitatSimActions.turn_right,
                ]
            )
        return {"action": self._adjust_action(action)}

    def memory_consolidation(self, info: Metrics):
        r"""This method is not used in this agent."""
        pass

    def _adjust_action(self, action):
        if (
                {action, self.action_history[-1]} == {HabitatSimActions.turn_left, HabitatSimActions.turn_right} or
                (
                        self.action_history[-1] != HabitatSimActions.move_forward and
                        self.action_history[:].count(self.action_history[-1]) == self.action_history.length
                )
        ):
            # avoid infinite loop of left<->right turns; move a bit after turning 90 degrees
            # the issue is relevant here because the sim is deterministic and action space discrete
            action = HabitatSimActions.move_forward
        self.action_history.update(action)
        return action


class GoalFollower(RandomAgent):
    def __init__(self, config) -> None:
        # config should be the object returned by habitat.get_config()
        super().__init__(config)
        turn_angle = config.habitat.simulator.turn_angle
        self.angle_th = float(np.deg2rad(turn_angle))

    @property
    def no_more_current_episode(self):
        return True

    def action_towards_goal(self, angle_to_goal: ndarray) -> int:
        if abs(angle_to_goal) < self.angle_th:
            action = HabitatSimActions.move_forward
        else:
            if angle_to_goal < 0:
                action = HabitatSimActions.turn_right
            else:
                action = HabitatSimActions.turn_left
        return action

    def follow_goal(self, observations: Observations):
        angle_to_goal = normalize_angle(observations['pointgoal_with_gps_compass'][1])
        action = self.action_towards_goal(angle_to_goal)
        return action

    def act(self, observations: Observations) -> Dict[str, int]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            action = self.follow_goal(observations)
        return {"action": self._adjust_action(action)}


class BugCX(GoalFollower):
    def __init__(self, config) -> None:
        # config should be the object returned by habitat.get_config()
        super().__init__(config)
        self.turn_angle = config.habitat.simulator.turn_angle
        # init in self.reset()
        self.escape_countdown = 0
        self.escape_action = None
        self.rotation_temp_goal = None
        self.reset()

    def reset(self):
        self.escape_countdown = 0
        self.escape_action = None
        self.rotation_temp_goal = None
        super().reset()

    def temp_goal(self, observations: Observations):
        angle_to_escape = observations['angle_to_escape']
        if angle_to_escape != 0:
            # collision detected
            self.escape_countdown = ((abs(angle_to_escape) - np.deg2rad(90)) // np.deg2rad(self.turn_angle)) * 2
            angle_to_temp_goal = angle_to_escape - np.deg2rad(90) * np.sign(angle_to_escape)
            self.rotation_temp_goal = angle_to_temp_goal + observations['rotation']
        else:
            # self.escape_countdown > 0 must be true
            self.escape_countdown -= 1
            angle_to_temp_goal = self.rotation_temp_goal - observations['rotation']
        return angle_to_temp_goal

    def avoid_obstacle(self, observations: Observations):
        angle_to_temp_goal = normalize_angle(self.temp_goal(observations))
        action = self.action_towards_goal(angle_to_temp_goal)
        return action

    def act(self, observations: Observations) -> Dict[str, int]:
        if self.is_goal_reached(observations):
            action = HabitatSimActions.stop
        else:
            if observations['angle_to_escape'] != 0 or self.escape_countdown > 0:
                action = self.avoid_obstacle(observations)
            else:
                action = self.follow_goal(observations)
        return {"action": self._adjust_action(action)}


class LateralisedMushroomBody:
    def __init__(self,
                 pn_shape,
                 n_pn_per_kc=10,
                 n_kc=32000,
                 sparsity_kc=0.01,
                 kc2mbon_learn_rate=1,
                 modulation_decay_rate=0.95):
        self.mb = MultiOutputMushroomBody(N_pn=np.prod(pn_shape),
                                          N_kc=n_kc,
                                          N_mbon=2,
                                          N_pn_perkc=n_pn_per_kc,
                                          S_kc=sparsity_kc,
                                          kc2mbon_rule='decay')
        self.learn_rate = kc2mbon_learn_rate
        self.decay_rate = modulation_decay_rate
        # init in self.reset()
        self.last_collision = None
        self.last_step_kc = None
        self.last_step_modulation = 0
        self.reset()

    def reset(self):
        self.last_collision = None
        self.last_step_kc = None
        self.last_step_modulation = 0

    def _get_kc_activity(self, observations: Observations):
        pn = observations['low_quality_view'].flatten()
        kc = self.mb.hashing(pn)
        return kc

    def modulation(self, observations: Observations):
        kc = self._get_kc_activity(observations)
        mbon = self.mb.evaluating(kc).flatten()
        modulation = (mbon[0] - mbon[1]) * pi
        self.last_step_modulation = self.last_step_modulation * self.decay_rate + modulation * (1 - self.decay_rate)
        return self.last_step_modulation

    def update_visual_memory(self, escape_turn_end, observations: Observations):
        kc = self._get_kc_activity(observations)
        angle_to_escape = observations['angle_to_escape']
        if angle_to_escape == 0 and not escape_turn_end:
            pass    # no collision-related learning
        else:
            if angle_to_escape < 0 or (escape_turn_end and self.last_collision == 'right'):
                # left is repulsive; right is attractive
                self.mb.learning(self.last_step_kc, [self.learn_rate, 0])
            elif angle_to_escape > 0 or (escape_turn_end and self.last_collision == 'left'):
                self.mb.learning(self.last_step_kc, [0, self.learn_rate])
            self.last_collision = 'left' if angle_to_escape < 0 else 'right'
        self.last_step_kc = kc

    @property
    def w_kc2mbon(self):
        return self.mb.kc2mbon.W_hash2val

    @w_kc2mbon.setter
    def w_kc2mbon(self, weight):
        self.mb.kc2mbon.W_hash2val = weight


class BugMBCX(BugCX):
    def __init__(self, config, mushroom_body: LateralisedMushroomBody, model_name=None) -> None:
        # config should be the object returned by habitat.get_config()
        self.last_soft_spl = 0.0
        self.mb = copy.deepcopy(mushroom_body)
        self.mb_weight_ITM = self.mb_weight_STM.copy()
        self.mb_weight_LTM = self.mb_weight_STM.copy()
        super().__init__(config)
        if model_name is not None: self.name = model_name
        self.reset()

    def reset(self):
        self.mb.reset()
        super().reset()

    def memory_consolidation(self, info: Metrics):
        soft_spl = info['spl']
        self.is_performance_improved = soft_spl >= self.last_soft_spl
        if self.is_performance_improved:
            self.mb_weight_LTM = self.mb_weight_ITM.copy()
            self.mb_weight_ITM = self.mb_weight_STM.copy()
        else:
            self.mb_weight_STM = self.mb_weight_LTM.copy()
            self.mb_weight_ITM = self.mb_weight_LTM.copy()
        self.last_soft_spl = soft_spl

    def follow_goal(self, observations: Observations):
        angle_to_goal = observations['pointgoal_with_gps_compass'][1]
        angle_to_modulated_goal = normalize_angle(angle_to_goal + self.mb.modulation(observations))
        action = self.action_towards_goal(angle_to_modulated_goal)
        return action

    def avoid_obstacle(self, observations: Observations):
        angle_to_temp_goal = self.temp_goal(observations)
        angle_to_modulated_goal = normalize_angle(angle_to_temp_goal + self.mb.modulation(observations))
        action = self.action_towards_goal(angle_to_modulated_goal)
        return action

    def act(self, observations: Observations) -> Dict[str, int]:
        self.mb.update_visual_memory(self.escape_countdown == 1, observations)
        action_dict = super().act(observations)
        return action_dict

    @property
    def mb_weight_STM(self):
        return self.mb.w_kc2mbon

    @mb_weight_STM.setter
    def mb_weight_STM(self, weight):
        self.mb.w_kc2mbon = weight

    @property
    def no_more_current_episode(self):
        is_memory_saturated = np.array_equal(self.mb_weight_LTM, self.mb_weight_ITM)
        return is_memory_saturated or not self.is_performance_improved
