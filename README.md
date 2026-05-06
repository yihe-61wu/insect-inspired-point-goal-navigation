# Insect-inspired Approach for Point-goal Navigation
## Overview
We develop a novel model for point-goal navigation by drawing an analogy between the formal benchmark of the Habitat point-goal navigation task and the ability of insects to discover, learn, and refine visually guided paths around obstacles between a discovered food location and their nest.
The model consists of two components, abstracted from two insect brain structures, the Mushroom Body (MB) responsible for (visual) associative learning and the Central compleX (CX) for path integration.

The MB-CX integrative model is tested in two simulators: Habitat and iGibson.
Habitat is ued for comparing to state-of-the-art models, while the more realistic physics engine of iGibson permits validation of the model's robustness against perturbation.

## How to Use
Due to the different physics engines, the model needs to be run in the two different simulators.
Two independet `conda` environments are recommended, because the two simulators require many common dependencies but of different versions, e.g. `OpenCV`.
We have been using two independent environments throughout development and testing on an `Ubuntu 22.04 LTS` laptop with `Python 3.8.13`.

Initial steps to install the two simulators are shown below, and more details can be found in the corresponding README files in the `Habitat` and `iGibson` directories, respectively.

### Installing Habitat
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

### Installing iGibson
Please install all dependencies required by [iGibson 2.0](https://stanfordvl.github.io/iGibson/installation.html) first.
Additionally with `numpy` and `matplotlib`, one can reproduce simulations and visualisations (given simulated data).

### Reconstruction Datasets
After installing both simulators, one should be able to test run some scripts but in very few scenes.
To fully replicate our work, at least the Habitat simulations in our paper, it is necessary to download the Gibson 4+ dataset, which requires an official license (free for research use).


## The Insect-inspired Model
### CX
The insect CX is a shallow neural circuit, playing a critical role in path integration, i.e. dead reckoning.
To better align with the Habitat challenge, we abstract the CX into a simple algorithm for GPS+Compass.

### MB
The insect MB, also a shallow circuit, is capable of rapid associative learning. The MB, as a visual memory in our models, is assumed to be a two-layered neural network, consisting of
- visual projection neurons (PN), receiving preprocessed visual inputs,
- Kenyon cells (KC), encoding any PN activity as a latent, sparse pattern, by multiplying the PN-KC weight matrix,
- MB output neurons (MBON), computing visual familiarity/novelty of the KC pattern, by multiplying the KC-MBON weight matrix.
- DopAminergic Neurons (DAN), activated by collision-related events and triggering learning.
where
- the PN-KC matrix is randomly initialised to be binary and sparse, and fixed throughout a simulated experiment; and
- the KC-MBON matrix is initialised to be one, as learning is achieved by synaptic depression (i.e., decreasing weights, which is more consistent with the real MB).

Other than the specific learning rule, the most important parameters of the MB are N_pn (the number of PN), N_pn_perkc (the sparsity of the PN-KC connectivity), N_kc (the number of KC) and S_kc (the sparsity of KC activity).

## Licenses
The MB-CX integrative model is different from the MB-only model in our previous work for visual route following,
but we reuse part of the code from [insect-inspired-route-following](https://github.com/yihe-61wu/insect-inspired-route-following) (under GNU GPL 3.0 license), particularly those for low-level perception and control in the iGibson simulators.

You are welcome to (re-)use everything in this repository, but please (re-)distribute any derivative works with the same GNU GPL 3.0 license and cite us:
1. Lu, Y., Cen, J., Alkhoury Maroun, R. et al. (2025). Insect-inspired Embodied Visual Route Following. _Journal of Bionic Engineering_. **22**, 1167–1193 [https://doi.org/10.1007/s42235-025-00695-8](https://doi.org/10.1007/s42235-025-00695-8)
2. Yihe, L., & Webb, B. (2026). An Efficient Insect-inspired Approach for Visual Point-goal Navigation. _[arXiv preprint arXiv:2601.16806](https://arxiv.org/abs/2601.16806)_.
