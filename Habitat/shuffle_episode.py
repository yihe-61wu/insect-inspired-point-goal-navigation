import os
import git
import numpy as np

import habitat
from habitat.core.dataset import EpisodeIterator
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)

repo = git.Repo(".", search_parent_directories=True)

dir_path = "/home/yihelu/habitat-lab"
os.chdir(dir_path)


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
    config.habitat.dataset.split = 'val'
# Create dataset
dataset = habitat.make_dataset(
    id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
)

episodes = dataset.episodes


scene_train = []
scene_test = []
# Iterate through all 71*14=994 episodes
for i, episode_train in enumerate(episodes[:]):
    # Process each episode
    # Your episode processing logic here
    scene_train.append(episode_train.scene_id)

    test_iterator = EpisodeIterator(
        episodes=episodes,
        shuffle=True,  # Whether to shuffle episodes
        group_by_scene=False,  # When shuffling episodes, whether to group them by scene
    )
    for j, episode_test in enumerate(test_iterator):
        if j >= 20: break
        scene_test.append(episode_test.scene_id)



print(i)
print(np.unique(scene_train, return_counts=True)[1])
print(np.unique(scene_test[:20], return_counts=True)[1].size)