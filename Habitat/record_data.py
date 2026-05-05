import copy
import os
import json
import numpy as np
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)


class TrialRecorder:
    def __init__(self, headless):
        self.headless = headless
        self.record = self.init_record()
        if not self.headless:
            self.vis_frames = []

    def init_record(self):
        record = {'trial_name': '',
                  'agent_name': '',
                  'scene_id': '',
                  'episode_id': '',
                  'start_position': [],
                  'goal_position': [],
                  'final_distance_to_goal': -1.0,
                  'success': 0.0,
                  'spl': 0.0,
                  'soft_spl': 0.0,                  
                  'collision_count': 0,
                  'position': [],
                  'rotation': [],
                  'pointgoal_with_gps_compass': [], # this is the distance angle diff to goal, not x, y position
                  'is_collision': [],
                  'angle_to_escape': [],
                  'collision_direction': []}
        return record

    def record_step(self, observations, info, additional=None):
        self.record['final_distance_to_goal'] = info['distance_to_goal']
        self.record['success'] = info['success']
        self.record['spl'] = info['spl']
        self.record['soft_spl'] = info['soft_spl']
        self.record['collision_count'] = info['collisions']['count']
        self.record['position'].append(observations['position'].tolist())
        self.record['rotation'].append(observations['rotation'].tolist())
        self.record['pointgoal_with_gps_compass'].append(observations['pointgoal_with_gps_compass'].tolist())
        self.record['angle_to_escape'].append(observations['angle_to_escape'].tolist())
        self.record['is_collision'].append(info['collisions']['is_collision'])
        self.record['collision_direction'].append(info['collisions']['collision_direction'].tolist())
        if not self.headless:
            # Concatenate visual observation and topdowm map into one image
            frame = observations_to_image(observations, info)
            # Remove data that do not show in video
            video_info = copy.deepcopy(info)
            video_info.pop("top_down_map")
            video_info['collisions'].pop("collision_direction")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, video_info, additional=additional)
            # Add fame to vis_frames
            self.vis_frames.append(frame)

    def record_start(self, start_goal_position, trial_name, observations, info):
        scene_id, episode_id, agent_name, trial_id = trial_name.split('_')
        self.record['trial_name'] = trial_name
        self.record['agent_name'] = agent_name
        self.record['scene_id'] = scene_id
        self.record['episode_id'] = episode_id
        self.record['trial_id'] = trial_id
        self.record['start_position'] = start_goal_position[0].flatten().tolist()
        self.record['goal_position'] = start_goal_position[1].flatten().tolist()
        self.record_step(observations, info)

    def record_end(self, output_path):
        # Create json file that stores trial data for future analysis
        data_file = os.path.join(output_path, f"{self.record['trial_name']}.json")
        with open(data_file, "w") as f:
            # Double check data type of self.record: should NEVER contain numpy arrays
            json.dump(self.record, f)
        # Create video from images and save to disk
        if not self.headless:
            images_to_video(
                self.vis_frames, output_path, self.record['trial_name'], fps=6, quality=10
            )
            self.vis_frames.clear()
        # Clear record
        self.record = self.init_record()
