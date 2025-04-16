# Topologic UntaNgling 2D mEsher

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/636e49252a1a4169b4db34b184522372)](https://app.codacy.com/gh/LIHPC-Computational-Geometry/tune/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/636e49252a1a4169b4db34b184522372)](https://app.codacy.com/gh/LIHPC-Computational-Geometry/tune/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The aim of this project is to provide an environment to implement 
Reinforcement Learning algorithms that aim to topologically
modify a 2D mesh. More specifically, we implement the work of 
*A. Narayanana*, *Y. Pan*, and *P.-O. Persson*, which is described 
in "**Learning topological operations on meshes with application to 
block decomposition of polygons**" (see [arxiv article](https://arxiv.org/pdf/2309.06484.pdf)
and [presentation](http://tetrahedronvii.cimne.com/slides/Persson.pdf)).

#### See the documentation website for more details: https://lihpc-computational-geometry.github.io/tune/

## Installation
The project can be cloned from github

## Usage 
The project can be used to train a reinforcement learning agent on **triangular meshes** or **quadrangular meshes**.

---

### Triangular Meshes

For training on triangular meshes, you can use an agent with all three actions: **flip**, **split**, and **collapse**. Two training models are available:

1. **Custom PPO Model** (`tune/model_RL/PPO_model`)
2. **PPO from Stable Baselines 3 (SB3)**

#### 🚀 Starting Training

##### 1. Using `tune/model_RL/PPO_model`

- Configure the model and environment parameters in:  
  `tune/training/train.py`

- Then run the following command from the `tune/` directory:
  ```bash
  python main.py
  ```


##### 2. Using PPO from Stable Baselines 3 (SB3)

- Configure the model and environment parameters in:  
    - `tune/environment/environment_config.json`
    - `tune/model_RL/parameters/PPO_config.json`

- Then run the training script in pycharm `tune/training/train_trimesh_SB3.py`

###### Flip-Only Training (SB3 PPO)

To train an agent using only the flip action with SB3 PPO, run the training script in pycharm `tune/training/train_trimesh_flip_SB3.py`

---

### Quadrangular Meshes

For training on quadrangular meshes, you can use an agent with all four actions: **flip clockwise**, **flip counterclockwise**, **split**, and **collapse**. Two training models are available:

1. **Custom PPO Model** (`tune/model_RL/PPO_model_pers`)
2. **PPO from Stable Baselines 3 (SB3)**

#### 🚀 Starting Training

##### 1. Configure the model and environment parameters in :
  - `tune/environment/environment_config.json`
  - `tune/model_RL/parameters/PPO_config.json`
  
##### 2. Using `tune/model_RL/PPO_model_pers`
Run the following command from the `tune/` directory:
  ```bash
  python -m /training/train_quadmesh
  ```

##### 3. Using PPO from Stable Baselines 3 (SB3)
Run the training script in pycharm `tune/training/train_quadmesh_SB3.py`



