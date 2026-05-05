# Insect-inspired Approach for Point-goal Navigation
## Overview
We develop a novel model for point-goal navigation by drawing an analogy between the formal benchmark of the Habitat point-goal navigation task and the ability of insects to discover, learn, and refine visually guided paths around obstacles between a discovered food location and their nest.
The model consists of two components, abstracted from two insect brain structures, the Mushroom Body (MB) responsible for (visual) associative learning and the Central compleX (CX) for path integration.

The model is tested in two simulators: Habitat and iGibson.
Habitat is ued for comparing to state-of-the-art models, while the more realistic physics engine of iGibson permits validation of the model's robustness against perturbation.

## How to Use
Due to the different physics engines of the two simulators, the model needs to be run in two different environments. Details can be found in the corresponding README files in the `Habitat` and `iGibson` directories, respectively.
