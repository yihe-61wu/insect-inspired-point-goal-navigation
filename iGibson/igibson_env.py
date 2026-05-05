import argparse
import logging
import os
import time
from collections import OrderedDict

import gym
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat

from igibson import object_states
from igibson.envs.env_base import BaseEnv
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.behavior_task import BehaviorTask
from igibson.tasks.dummy_task import DummyTask
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW

from igibson.envs.igibson_env import iGibsonEnv

log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    args.config = "/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/configs/locobot_nav.yaml"
    args.mode = "gui_interactive"

    env = iGibsonEnv(config_file=args.config, mode=args.mode, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(1):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        keys_state = 'task_obs', 'rgb', 'depth'
        data = {}
        for key in keys_state:
            data[key] = []
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            # print("state", state.keys()) ## state odict_keys(['task_obs', 'rgb', 'depth']) shape: (4,), (90,160,3), (90,160,1)
            [data[key].append(state[key]) for key in keys_state]
            print("reward", reward)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))

        for key in keys_state:
            print(np.shape(data[key]))
    env.close()
