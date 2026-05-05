import os
from typing import TYPE_CHECKING, Union, cast
import git

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat_sim.utils import viz_utils as vut

from agents import ShortestPathFollowerAgent, RandomAgent, GoalFollower
from record_data import TrialRecorder
from habitat_api import EnvStateOracle

import matplotlib.pyplot as plt


# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

repo = git.Repo(".", search_parent_directories=True)
dir_path = "/home/yihelu/habitat-lab"
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(
    repo.working_tree_dir, "examples/tutorials/"
)
os.makedirs(output_path, exist_ok=True)
os.chdir(dir_path)



#####################
# Create habitat config
config = habitat.get_config(
    config_path=os.path.join(repo.working_tree_dir, "config/pointnav_rgb.yaml")
)
# Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
with habitat.config.read_write(config):
    config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        }
    )
# Create dataset
dataset = habitat.make_dataset(
    id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
)

with habitat.Env(config=config, dataset=dataset) as env:
    topdown_map = maps.get_topdown_map_from_sim(env.sim)
    plt.imshow(topdown_map)
    plt.show()