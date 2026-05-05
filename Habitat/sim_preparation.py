import os
import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)

def get_pointnav_config(git_repo):
    # Change to the habitat-lab directory
    # This is necessary to ensure that the habitat config files are found correctly
    # If you are running this script from a different directory, adjust the path accordingly
    dir_path = "/home/yihelu/habitat-lab"
    os.chdir(dir_path)

    # Create habitat config
    config = habitat.get_config(
        config_path=os.path.join(git_repo.working_tree_dir, "config/pointnav_rgb.yaml")
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
    return config, dataset

