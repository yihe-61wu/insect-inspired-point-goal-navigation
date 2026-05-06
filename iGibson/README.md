# Insect Mushroom Body Models for Mapless Visual Locomotion 
## Overview
For completeness and reproducibility, this repository contains all files of functions, objects, scripts (mostly as `*.py` files)
and analysis (mostly as `*.ipynb` files),
including those for testing, explorative or comparison purposes and even some to be deprecated in the future.
This repository does not contain simulation data due to their synthetic nature and excessive quantity.


## How to use
### Installation and dependencies
Please install all dependencies required by [iGibson 2.0](https://stanfordvl.github.io/iGibson/installation.html) first.
Additionally with `numpy` and `matplotlib`, one can reproduce simulations and visualisations (given simulated data).

A `conda` virtual environment is recommended, because
1. we have been using it throughout development and testing, and
2. iGibson is built on various common packages that a user might rely on in other projects, such as `PyBullet` and `OpenCV`.

All the code has been developed and tested on an `Ubuntu 22.04 LTS` laptop 
in a `Python 3.8.13` environment with all packages specified in `requirements.txt`.


### List of main files
  * `\yaml_robots\*.yaml` - the robot parameters,
  * `igibson_api.py` - the functions and objects interfacing our models and the simulator,
  * `neuro_utils.py` - the common, basic objects for building MB circuits,
  * `robot_sensory.py` - the common, basic objects for sensory inputs to the models,
  * `simulation_main.py` - the common script for initialising simulation, robots and models, with respect to various keyword parameters,
  * `video_recorder.py` - the object for recording data,
  * `analysis` in a file name - indicates the file is for data analysis and visualisation.
  * `goal_directed_navigation.py` - the script for the experiment of goal-directed locomotion,
  * `goal_directed_navigation_2robots.py` - the script for the experiment of two robots running towards each other,
  * `bug_MBCX.py` - insect-inspired models (including full and ablated ones)
  * `mushroom_body_new.py` - MB (with variations) in used in our model.

Note there are redundancies in the code,
e.g., `mushroom_body.py` and `mushroom_body_new.py` can be merged,
but we have not removed all of them to minimise the risk of incompatibility caused by code refactoring and to maximise reproducibility.

### Running simulations
In addition to the scripts mentioned in the previous section for the main navigation tasks, 
there are auxillary scripts such as `examine_robot_property.py` available for secondary purposes.
Nevertheless, these scripts share the same workflow:
1. **Importing modules**: While all the simulations are to take place within the simulator, iGibson, 
most of the scripts do not import iGibson modules explicitly,
because the necessary iGibson modules are integrated in `igibson_api.py`,
which is imported into other files for model configuration, etc.
_Remark: double importing iGibson modules would typically lead to running errors._
2. **Preparing simulation, model(s) and recorder(s)**: Preparing simulation is composed of a sequence of typical steps of 
initialising simulator, scene and robot(s), 
which are wrapped under `SimulationOrganisor` in `igibson_api.py`.
Depending on the truth value of `headless`, viewers showing live videos for human users are to be additionally initialised.
The steps of configuration and initialisation of our models are wrapped in `simulation_main.py`,
and the steps for the data recorder under `Recorder` in `video_recorder.py`.
_Remark: each `Recorder` object can record data from only a single robot (the one under the control of an MB model)._
3. **Repeating trials**: In the route following experiments, training and test trials are separated. 
In the goal-directed locomotion experiments, online learning is assumed within a trial 
and memory is inherited across trials.
Within every trial, the route is to firstly put the robot back at the starting location
and start a new session for data recording,
then to begin the simulation, a loop in which robot and model states are varied and recorded,
and finally to terminate the loop when the robot reaches its target position or the time is up,
followed by saving the recorded data to files (and generating a video if `headless==True`) and starting a new loop (trial).
4. **Ending simulation**: The simulator is to be disconnected, and auxillary figures generated.
_Remark: The disconnection of the simulator is essential, 
especially when a user attempts to repeat simulations by running the `main` function of a script.
Failure to do so would rapidly overload computer memories.
In contrast, the generation of the auxillary figures is efficient but not essential._
                                         
We recommend `headless` to be set as `False` when running early-stage simulations or producing video demos,
but as `True` when running repetitive simulations (for data analysis),
because showing, recording and saving live videos add a significant amount of computational overhead.

### Configuring simulator, scene and robot(s)
A user can customise properties of simulator, scene and robot(s) by specifying `**kwargs`
for the `main` function of a script.
The values are passed to `kwargs_main`.
If not specified, default values in `simulation_main.py` are used.

Typically in our main scripts, default values for the iGibson simulator (its physics engine and visual renderer) are used.

There are multiple indoor interactive scenes available from iGibson.
The choice of a scene can be specified by setting `scene_name`.
If `scene_name=='random'`, a scene will be chosen randomly.

There are multiple robot models available from iGibson.
We have considered only the ones under `\yaml_robots`, because they are all differential wheeled robots with a front camera.
The properties of sensory input and motor control can be customised such as their noise level.

Moving robots (under manual control) can be added as dynamical obstacles 
by `SimulationOrganisor.prepare_dynamical_obstacles`.
An example can be found in `goal_directed_navigation.py` (commented out by default).

If a user needs to modify properties beyond the existing ones in `kwargs_main`,
they should familiarise themselves with the underlying iGibson packages, the `*.yaml` files and our models.
 
### Customising models
See the **Main models** section.

### Reproducing data analysis and visualisation
Most of our data analysis and visualisation have been conducted with the `*.ipynb` files in jupyter notebook.
We note, however, `jupyter` is delibrately omitted from the dependency requirements for running simulation,
because we use a different `conda` virtual environment for data analysis.

In addition, we do not provide our data due to their synthetic nature.
By default, all the simulation scripts will save data under `\records`.
A user trying to reuse the `*.ipynb` files is recommended to pay attention to their directory paths.

## Main models
### Structure
A full model consists of modules from, in a bottom-up order:
* `utils.py` and `igibson_api` - generic functions and simulator API,
* `robot_sensory.py` and `neuro_utils.py` - basic components for sensory inputs and MB circuit,
* `mushroom_body_new.py` - MB circuits (for learning only),
* `robot_steering.py` and `bug_MBCX.py` - full MB models (from sensory inputs to control outputs)
for goal-directed locomotion.

We note there exists a large amount of redundancy between `robot_steering.py` and `bug_MBCX.py`.
This is partially intentional, because they are developed for two different models
(despite their similarity in visual learning and memory).

### Mushroom body (MB) structure
Insect MB, a shallow circuit, is capable of rapid associative learning.
The MB, as a visual memory in our models, is assumed to be a two-layered neural network,
consisting of 
- visual projection neurons (PN), receiving preprocessed visual inputs,
- Kenyon cells (KC), encoding any PN activity as a latent, sparse pattern, by multiplying the PN-KC weight matrix,
- MB output neurons (MBON), computing visual familiarity/novelty of the KC pattern, by multiplying the KC-MBON weight matrix.
      
where 
- the PN-KC matrix is randomly initialised to be binary and sparse, and fixed throughout a simulated experiment; and
- the KC-MBON matrix is initialised to be 
  - either zero, if learning is achieved by synaptic enhancement (i.e., increasing weights),
  - or one, if learning is achieved by synaptic depression (i.e., decreasing weights, which is more consistent with the real MB).

The MB circuit (with all the variations) can be found in `mushroom_body.py` and `mushroom_body_new.py`.
Other than the specific learning rule, the most import parameters of these objects are 
`N_pn` (the number of PN), `N_pn_perkc` (the sparsity of the PN-KC connectivity), `N_kc` (the number of KC) and `S_kc` (the sparsity of KC activity).

### MB as a high-level modulator
The model here is different from our previous work for  [insect-inspired-route-following](https://github.com/yihe-61wu/insect-inspired-route-following),
as the MB uses visual memory to modulate the desired heading direction generated by the CX (performing PI),
and its learning is triggered only by collision escape events.
As a result, the model is expected to learn visual cues of obstacles, 
and eventually avoid collisions in a predictive manner.

The variations of this model can also be found in `simulation_main.py`, 
while the core modules are specified in `bug_MBCX.py`.
Running `goal_directed_navigation.py` simulates these models.
We highlight an online learning paradigm is applied in this second model,
so the script does not have separated training and test sessions.

