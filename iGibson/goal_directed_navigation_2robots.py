import logging
import numpy as np
from simulation_main import *
from video_recorder import Recorder
from plot_from_data import DataVisualiser

from bug_MBCX import *
from mushroom_body_new import MultiOutputMushroomBody

def main(model_name='lamb',
         robot_name='scaled_freight',
         scene_name='random',
         save_dir='test',
         headless=True,
         **kwargs):
    ### initialisation
    for key, value in kwargs.items():
        kwargs_main[key] = value
    simulation1 = SimulationOrganisor(scene_name, robot_name, headless, mindist_start2goal=kwargs_main['mindist_start2goal'])
    robot1 = simulation1.prepare_robot()
    scene = simulation1.prepare_scene(load_object_categories=kwargs_main['load_object_categories'],
                                      texture_randomization=True)
    s = simulation1.prepare_simulator()
    simulation1.initialise_pointgoal_simulation(max_trial_time=900, fix_pos_goal=kwargs_main['fix_pos_goal'])

    simulation2 = SimulationOrganisor(scene_name, robot_name, headless, mindist_start2goal=kwargs_main['mindist_start2goal'])
    robot2 = simulation2.prepare_robot()
    simulation2.scene = scene
    simulation2.simulator = s
    simulation2.initialise_pointgoal_simulation2(max_trial_time=900, fix_pos_goal=np.flipud(kwargs_main['fix_pos_goal']))

    model_names = 'MB4ONaxial', 'MB4ONdiagonal', 'MB2ONbilateral', 'MB2ONopponent'
    toy_brains1 = [construct_MBCX_model(model_name, simulation1, **kwargs_main) for model_name in model_names]
    toy_brains2 = [construct_MBCX_model(model_name, simulation2, **kwargs_main) for model_name in model_names]

    var_monitor = ['vl', 'va', 'steer_mode']
    recorder1 = Recorder(simulation1, save_dir, *var_monitor)
    recorder2 = Recorder(simulation2, save_dir, *var_monitor)

    N_trial = 10
    for trial in range(N_trial):
        print('trial {}'.format(trial))
        for model_name, toy_brain1, toy_brain2 in zip(model_names, toy_brains1, toy_brains2):
            print('model {}'.format(model_name))

            simulation1.reinit_robot_pose(rand_orn=True, cooling_time=1, add_moving_obstacle=False)
            simulation2.reinit_robot_pose(rand_orn=True, cooling_time=1, add_moving_obstacle=False)
            # simulator.reload maybe needed
            toy_brain1.reset()
            toy_brain2.reset()
            recorder1.start_recording()
            recorder2.start_recording()

            for t_ctrl in range(simulation1.max_trial_time):
                vl, va = toy_brain1.steer()
                robot1.apply_action((vl, va))

                recorder1.recording(vl=vl, va=va, steer_mode=toy_brain1.steer_mode[-1])
                dist2goal1 = np.linalg.norm(robot1.get_position()[:2] - simulation1.pos_goal[:2])

                vl, va = toy_brain2.steer()
                robot2.apply_action((vl, va))

                recorder2.recording(vl=vl, va=va, steer_mode=toy_brain1.steer_mode[-1])
                dist2goal2 = np.linalg.norm(robot2.get_position()[:2] - simulation2.pos_goal[:2])

                s.step()

                goal_reached = dist2goal1 <= simulation1.robot_radius and dist2goal2 <= simulation2.robot_radius
                if goal_reached: break

            # toy_brain.trial_end_learn(goal_reached)

            robot1.keep_still()
            robot2.keep_still()
            take_idx = '{}1_trial_{}'.format(model_name, trial)
            recorder1.stop_recording(take_idx)
            take_idx = '{}2_trial_{}'.format(model_name, trial)
            recorder2.stop_recording(take_idx)


    s.disconnect()

    var_names = var_monitor
    # var_names = []
    painter = DataVisualiser(recorder1.save_dir, headless)
    painter.draw_summary(var_names)

    painter = DataVisualiser(recorder2.save_dir, headless)
    painter.draw_summary(var_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save_dir = 'tworobots'
    for _ in range(100):
        main(save_dir=save_dir,
             robot_name='scaled_freight',   #'freight',  #
             headless=True,
             scene_name='Pomaria_1_int',
             # scene_name='random', #'stadium', #
             fix_pos_goal=([-6, 3, 0], [-6, 1, 0]),
             # load_object_categories='with_no_obj',
             mindist_start2goal=5,
             preprocess='di')
