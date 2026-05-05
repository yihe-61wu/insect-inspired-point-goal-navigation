# How to use
## Installing dependencies
1. Install [`habitat-sim`](https://github.com/facebookresearch/habitat-sim) (v0.3.1):
   1. Following README instructions for the `conda` option, but ignoring everything including and after **Testing**.
   2. Uninstall and (re)install `numpy` (v1.23) using `pip`, due to its compatiblitiy with `habitat-lab`.
2. Install [`habita-lab`](https://github.com/facebookresearch/habitat-lab) (v0.3.1):
   1. Following README instructions for the `conda` option, but not repeating the first steps as `habitat-sim` has been installed.
   2. Double-checking the versions of `habitat-sim` and `habitat-lab`, which should be the same.
3. Download assets under `habitat-lab/data` by `git lfs pull`.
   1. The download commands executed when installing `habitat-sim` and `habitat-lab` did not download the data; they only create pointers.
   2. At this stage, the jupyter notebooks under `habitat-lab/examples` should work.
   3. However, to make python files run, there is a conflict between `opencv-python` and `PyQt5`. The solution is to uninstall `opencv-python` and install `opencv-python-headless` instead, using `pip`.

## Configuration
The `*.yaml` files under `config/` are used to configure navigation tasks and agents.
Note there exist dependencies amongst them.
In particular, `pointnav_base_test.yaml` imports `pointnav_base.yaml` and `habitat_test.yaml`.

`habitat_test.yaml` uses seemingly randomised start and goal positions, which are actually determined by the corresponding `*.json.gz` file specified in line 9.

More details can be found under `habitat-lab/habitat-lab/habitat/config/` (one of this repo's dependencies).


## Data Recording
`TrialRecorder` in `record_data.py` is used to record data of individual trials for future analysis.
If `headless=False`, videos will be generated.


## Model details
### Agents
Any agent has 4 optional actions: 
- `move_forward`: the agent moves forward by 0.2 meter.
- `turn_left`/`turn_right`: the agent rotates by 10 degrees. 
- `stop`: The simulation trial is to be terminated at this epoch.

As such, `stop` is a special action, and there are 3 locomotion actions.

We provide 5 agents in `agents.py`:
- `ShortestPathFollowerAgent` has the oracle knowledge of the shortest path between the start and goal positions.
- `RandomAgent` chooses a random locomotion action at every epoch.
- `GoalFollower` uses dead-reckoning (only) to move towards the goal.
- `BugCX` uses dead-reckoning (inheriting `GoalFollower`), and when it detects a collision, it switches into an escape mode for a few epochs in the hope of avoiding the obstacle.
- `BugMBCX` inherits `BugCX`'s dead-reckoning and obstacle avoidance abilities, and additionally learns the valence of the key views after collisions.

### Collision
By default, `pointnav_base.yaml` sets `enable_physics = False`, which sacrifices simulation realism for determinism and efficiency. 
An agent in such environment is assumed to slide against an obstacle, 
and the simulator only generates a binary signal of collision, using `env.get_metrics['collisions']['is_collision]`.
This signal is not enough for our MB model, which requires a left or right signal, when collision occurs.

We thus create `EnvWrapper` and make estimations of collisions with its `augment_observations_info` method.
As an agent will slide against an obstacle following the obstacle's geometric boundary without changing its pose,
we estimate the force of a collision by the discrepancy between its actual and intended displacement.

We tried a different approach that estimated the force by approximating acceleration from velocity approximated from displacements,
which failed to provide reasonable results, probably due to the discrete nature of the simulation.
We decided not to set `enable_physics = True` for the simulation determinism and efficiency.