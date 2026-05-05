import os
import git
import numpy as np
import habitat
from habitat.core.dataset import EpisodeIterator

from habitat_api import EnvWrapper
from sim_preparation import get_pointnav_config
from agents import ShortestPathFollowerAgent, RandomAgent, GoalFollower, BugCX, BugMBCX, LateralisedMushroomBody
from record_data import TrialRecorder

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def pointgoal_navigation_trial(envw, current_episode, agent, recorder, trial, trial_path, headless):
    trial_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}_{agent.name}_{trial}"
    print(trial_name)
    # Set the new trial to use the specific episode
    envw.env.current_episode = current_episode

    agent.reset()
    observations, info = envw.env_reset()
    start_goal_position = envw.get_start_goal_position()
    recorder.record_start(start_goal_position, trial_name, observations, info)

    # Repeat the steps above while agent doesn't reach the goal
    while not envw.env.episode_over:
        # Get the next best action
        action = agent.act(observations)
        if action is None:
            break

        # Step in the environment
        observations, info = envw.env_step(action)
        # Record data
        # extra_info = [
        #               f"MB modulation: {np.rad2deg(agent.mb.last_step_modulation)}",
        #               ]
        recorder.record_step(observations, info,
                                # additional=extra_info
                                )

    recorder.record_end(trial_path)

    agent.memory_consolidation(info)


def pointgoal_navigation(num_train_episodes, max_train_trials, num_test_per_episode, output_path='test', headless=True):
    """
    :param num_train_episodes: Total number of episodes. A pointnav episode is defined by scene and start-goal positions.
    :param max_train_trials: Number of repeated trials per episode. An agent may preserve memory across trials if permitted by
        agent.reset()
    :param headless: True - no visuals to be displayed. On average, headless=False would triple the simulation time.
    """
    # Define the output path
    repo = git.Repo(".", search_parent_directories=True)
    output_path = os.path.join(repo.working_tree_dir, 'examples', output_path)
    os.makedirs(output_path, exist_ok=True)
    # Get the pointnav config and dataset
    config, dataset = get_pointnav_config(repo)
    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        envw = EnvWrapper(env)
        # Create TrialRecorder for videos
        recorder = TrialRecorder(headless)
        for episode_train in dataset.episodes[471:num_train_episodes]:
            # Create agent
            oracle = ShortestPathFollowerAgent(
                env=env,
                config=config,
            )
            mb = LateralisedMushroomBody(
                pn_shape=envw.view_shape,
                kc2mbon_learn_rate=1,
                modulation_decay_rate=0.95
            )
            bug_mbcx = BugMBCX(
                config=config,
                mushroom_body=mb
            )

            goal_follower = GoalFollower(
                config=config
            )
            bug_cx = BugCX(
                config=config
            )

            agents = [goal_follower, bug_cx, bug_mbcx]

            trial_path = os.path.join(output_path, f"{os.path.basename(episode_train.scene_id)}_{episode_train.episode_id}")
            os.makedirs(trial_path, exist_ok=False) # this will fail if the directory already exists, which is what we want

            for agent in agents:
                for i in range(max_train_trials):
                    pointgoal_navigation_trial(envw, episode_train, agent, recorder, f"train{i}", trial_path, headless)
                    # terminate the current episode immediately
                    # this happens for all the models except BugMBCX
                    if agent.no_more_current_episode: break

            # Create test episodes
            # Only BugMBCX-learnt needs to be tested, 
            # because all test episodes are used for other models (in episode_train)
            test_iterator = EpisodeIterator(
                episodes=dataset.episodes,
                shuffle=True,  # Whether to shuffle episodes
                group_by_scene=False,  # When shuffling episodes, whether to group them by scene
            )
            for j, episode_test in enumerate(test_iterator):
                if j >= num_test_per_episode: break
                bug_mbcx_learnt = BugMBCX(config=config, mushroom_body=bug_mbcx.mb, model_name="BugMBCXlearnt")
                pointgoal_navigation_trial(envw, episode_test, agent, recorder, f"test{j}", trial_path, headless)



if __name__ == "__main__":
    # import time
    # time_start = time.time()

    headless = True
    num_train_episodes = 1000
    max_train_trials = 20
    num_test_per_episode = 20
    pointgoal_navigation(
        num_train_episodes,
        max_train_trials, 
        num_test_per_episode, 
        output_path='all-episodes',
        headless=headless
        )

    # time_end = time.time()
    # print(f"mean trial time (real-world): {(time_end - time_start) / num_train_episodes / max_train_trials}")