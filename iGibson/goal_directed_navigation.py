import logging
import numpy as np
from simulation_main import *
from video_recorder import Recorder
from plot_from_data import DataVisualiser

from bug_MBCX import *
from mushroom_body_new import MultiOutputMushroomBody
from igibson_api import NoisyDDController

def main(model_name='lamb',
         robot_name='scaled_freight',
         scene_name='random',
         save_dir='test',
         headless=True,
         **kwargs):
    ### initialisation
    for key, value in kwargs.items():
        kwargs_main[key] = value
    simulation = SimulationOrganisor(scene_name, robot_name, headless, mindist_start2goal=kwargs_main['mindist_start2goal'])
    robot = simulation.prepare_robot()
    scene = simulation.prepare_scene(load_object_categories=kwargs_main['load_object_categories'])
    s = simulation.prepare_simulator()
    # add moving robot as dynamic obstacle
    # dynamic_obstacle = simulation.prepare_dynamical_obstacles(['freight'], [[-3, 1.2, 0]])
    simulation.initialise_pointgoal_simulation(max_trial_time=500, fix_pos_goal=kwargs_main['fix_pos_goal'])
    print(simulation.optimal_route)

    model_names = 'CXonly', 'MB2ONbilateral', 'GoalFollower', #'MB4ONaxial', 'MB4ONdiagonal', #'MB2ONopponent'
    toy_brains = [construct_MBCX_model(model_name, simulation, **kwargs_main) for model_name in model_names]
    noisyDDcontroller = NoisyDDController(robot, noise=kwargs_main['noise_motor'])

    var_monitor = ['vl', 'va', 'steer_mode']
    recorder = Recorder(simulation, save_dir, *var_monitor)

    N_trial = 50
    for trial in range(N_trial):
        print('trial {}'.format(trial))
        for model_name, toy_brain in zip(model_names, toy_brains):
            if trial > 0 and model_name in ('CXonly', 'GoalFollower'):
                continue

            print('model {}'.format(model_name))

            simulation.reinit_robot_pose(rand_orn=True, cooling_time=1)
            # simulator.reload maybe needed
            toy_brain.reset()
            recorder.start_recording()

            for t_ctrl in range(simulation.max_trial_time):
                vlva = toy_brain.steer()
                vl, va = noisyDDcontroller.apply_action(*vlva, noisefree=False)

                s.step()
                recorder.recording(vl=vl, va=va, steer_mode=toy_brain.steer_mode[-1])

                dist2goal = np.linalg.norm(robot.get_position()[:2] - simulation.pos_goal[:2])

                goal_reached = dist2goal <= simulation.robot_radius
                if goal_reached: break

            # toy_brain.trial_end_learn(goal_reached)

            robot.keep_still()
            take_idx = '{}_trial_{}'.format(model_name, trial)
            recorder.stop_recording(take_idx)


    s.disconnect()

    var_names = var_monitor
    # var_names = []
    painter = DataVisualiser(recorder.save_dir, headless)
    painter.draw_summary(var_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save_dir = 'point-goal-random-50'
    # for vab in (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5):
    # for noise_motor in (1, 0.5, 0.2, 0.01, 0.05, 0.02, 0.01):
    for _ in range(400):
        main(save_dir=save_dir,
             robot_name='scaled_freight',   #'freight',  #
             headless=True,
             scene_name='random',#'Pomaria_1_int', #'Beechwood_0_int', #'Wainscott_0_int',
             # scene_name='random', #'stadium', #
             # fix_pos_goal=([-6, 2, 0], [-1, 0, 0]),
             # fix_pos_goal=([-7.05, 3.39, 0], [-1.19, -3.14, 0]),
             # noise_motor=0.05,
             # va_bias=vab,
             # load_object_categories='with_no_obj',
             mindist_start2goal=10,
             preprocess='id')
