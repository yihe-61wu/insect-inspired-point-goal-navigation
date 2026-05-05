# Insect-inspired Approach for Point-goal Navigation
## Overview
We develop a novel model for point-goal navigation by drawing an analogy between the formal benchmark of the Habitat point-goal navigation task and the ability of insects to discover, learn, and refine visually guided paths around obstacles between a discovered food location and their nest.
The model consists of two components, abstracted from two insect brain structures, the Mushroom Body (MB) responsible for (visual) associative learning and the Central compleX (CX) for path integration.

The MB-CX integrative model is tested in two simulators: Habitat and iGibson.
Habitat is ued for comparing to state-of-the-art models, while the more realistic physics engine of iGibson permits validation of the model's robustness against perturbation.

## How to Use
Due to the different physics engines of the two simulators, the model needs to be run in the two different environments. Details can be found in the corresponding README files in the `Habitat` and `iGibson` directories, respectively.

## Model
### CX for path integration


### MB for visual learning and memory
Insect MB, a shallow circuit, is capable of rapid associative learning. The MB, as a visual memory in our models, is assumed to be a two-layered neural network, consisting of
- visual projection neurons (PN), receiving preprocessed visual inputs,
- Kenyon cells (KC), encoding any PN activity as a latent, sparse pattern, by multiplying the PN-KC weight matrix,
- MB output neurons (MBON), computing visual familiarity/novelty of the KC pattern, by multiplying the KC-MBON weight matrix.
- DopAminergic Neurons (DAN), activated by collision-related events and triggering learning.
where
- the PN-KC matrix is randomly initialised to be binary and sparse, and fixed throughout a simulated experiment; and
- the KC-MBON matrix is initialised to be one, as learning is achieved by synaptic depression (i.e., decreasing weights, which is more consistent with the real MB).

Other than the specific learning rule, the most important parameters of these objects are N_pn (the number of PN), N_pn_perkc (the sparsity of the PN-KC connectivity), N_kc (the number of KC) and S_kc (the sparsity of KC activity).

## Licenses
The MB-CX integrative model is different from the MB-only model in our previous work for visual route following,
but we reuse part of the code from [insect-inspired-route-following](https://github.com/yihe-61wu/insect-inspired-route-following) (under GNU GPL 3.0 license), particularly those for low-level perception and control in the iGibson simulators.

You are welcome to (re-)use everything in this repository, but please (re-)distribute any derivative works with the same GNU GPL 3.0 license and cite us:
1. Lu, Y., Cen, J., Alkhoury Maroun, R. et al. (2025). Insect-inspired Embodied Visual Route Following. _Journal of Bionic Engineering_. **22**, 1167–1193 [https://doi.org/10.1007/s42235-025-00695-8](https://doi.org/10.1007/s42235-025-00695-8)
2. Yihe, L., & Webb, B. (2026). An Efficient Insect-inspired Approach for Visual Point-goal Navigation. _[arXiv preprint arXiv:2601.16806](https://arxiv.org/abs/2601.16806)_.
