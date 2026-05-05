import numpy as np
import cv2

import habitat
from habitat_sim.utils.common import quat_to_angle_axis
from habitat.core.simulator import Observations
from habitat.core.embodied_task import Metrics

from utils import normalize_angle


def apply_lateral_inhibition(image, sigma_center, sigma_surround):
    """
    Apply lateral inhibition using the Difference of Gaussians (DoG) model.

    Args:
        image (numpy.ndarray): Input image
        sigma_center (float): Standard deviation of the center (excitatory) Gaussian kernel
        sigma_surround (float): Standard deviation of the surround (inhibitory) Gaussian kernel

    Returns:
        numpy.ndarray: Image with lateral inhibition applied
    """
    # Create Gaussian kernels
    size = int(max(sigma_center, sigma_surround) * 6) + 1
    center_kernel = cv2.getGaussianKernel(size, sigma_center)
    surround_kernel = cv2.getGaussianKernel(size, sigma_surround)

    # Create 2D Gaussian kernels
    center_kernel = center_kernel * center_kernel.T
    surround_kernel = surround_kernel * surround_kernel.T

    # Apply DoG model
    center_response = cv2.filter2D(image, -1, center_kernel)
    surround_response = cv2.filter2D(image, -1, surround_kernel)
    inhibited_image = center_response - surround_response

    return inhibited_image


class EnvWrapper:
    def __init__(self,
                 env: habitat.Env,
                 view_shape=(33, 33),
                 view_preprocess='id',  # 'id' is consistent with the model in the igibson project
                 change_reference_frame=True):
        rgb_config = env.sim.sensor_suite.sensors['rgb'].config
        self.frame_shape = rgb_config['height'], rgb_config['width']
        self.view_shape = view_shape
        self.view_preprocess = view_preprocess
        if 'i' in view_preprocess: self.sigma12 = 1, 2
        self.env = env
        self.forward_step_size = env._config.simulator.forward_step_size
        self.change_reference_frame = change_reference_frame
        self.prev_position = None   # init in self.reset()
        self.reset()

    def reset(self):
        self.prev_position, _ = self.get_position_rotation()

    # visual preprocessing methods
    def _rgb2view(self, rgb):
        view = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for pp in self.view_preprocess:
            view = self._preprocess_view(view, pp)
        return view

    def _preprocess_view(self, view, transform):
        if transform == 'i':
            view = apply_lateral_inhibition(view, *self.sigma12)
        elif transform == 'd':
            view = cv2.resize(view, self.view_shape)
        return view

    # kinematics methods
    def _change_rotation_ref(self, rotation):
        if self.change_reference_frame: rotation += np.pi
        return rotation

    def _change_position_ref(self, position):
        if self.change_reference_frame: position = np.flip(position)
        return position

    def get_start_goal_position(self):
        start = np.array(self.env.current_episode.start_position)[[0, 2]]
        goal = np.array(self.env.current_episode.goals[0].position)[[0, 2]]
        return self._change_position_ref(start), self._change_position_ref(goal)

    def get_position_rotation(self):
        # env.sim.get_agent_state().position returns [x, y, z]
        # Coordinate frame conventions: y-up and right-handed
        # x+: out from screen, y+: up, z+: left
        position =  self.env.sim.get_agent_state().position[[0, 2]]
        quat = self.env.sim.get_agent_state().rotation
        # axis is either [0, 1, 0] (up) or [0, -1, 0] (down)
        angle, axis = quat_to_angle_axis(quat)
        # rotation  = 0 aligns with x+
        # This rotation angle + pointgoal_with_gps_compass[1] = angle to goal
        rotation = angle * axis[1]
        return self._change_position_ref(position), self._change_rotation_ref(rotation)

    def _get_collision_direction(self, position, rotation, info: Metrics):
        if info['collisions']['is_collision']:
            # Estimate collision direction by the discrepancy between the actual and intended displacements
            # This method works only when enable_physics=False as the agent simply slides along the obstacle collided
            displacement = position - self.prev_position
            intended_displacement = np.array([np.cos(rotation), np.sin(rotation)]) * self.forward_step_size
            collision_direction = displacement - intended_displacement
            collision_rotation = np.arctan2(*np.flip(collision_direction))
            angle_to_escape = normalize_angle(collision_rotation - rotation)
        else:
            collision_direction = np.full(2, None)
            angle_to_escape = np.array(0.0)
        return collision_direction, angle_to_escape

    def augment_observations_info(self, observations: Observations, info: Metrics):
        position, rotation = self.get_position_rotation()
        collision_direction, angle_to_escape = self._get_collision_direction(position, rotation, info)
        low_quality_view = self._rgb2view(observations['rgb'])
        # augment dicts
        observations['low_quality_view'] = low_quality_view[:, :, np.newaxis]
        observations['low_quality_frame'] = cv2.resize(low_quality_view, self.frame_shape)[:, :, np.newaxis]
        observations['position'] = position
        observations['rotation'] = rotation
        observations['angle_to_escape'] = angle_to_escape
        info['collisions']['collision_direction'] = collision_direction
        # update state
        self.prev_position = position
        return observations, info

    def env_reset(self):
        observations = self.env.reset()
        info = self.env.get_metrics()
        self.reset()
        return self.augment_observations_info(observations, info)

    def env_step(self, action):
        # Step in the environment
        observations = self.env.step(action)
        info = self.env.get_metrics()
        return self.augment_observations_info(observations, info)

