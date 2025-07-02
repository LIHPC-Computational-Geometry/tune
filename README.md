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
  `tune/training/config/trimesh_config_PPO_perso.yaml`

- Then run the following command from the `tune/` directory:
  ```bash
  python -m training.train_trimesh.py
  ```
  
##### 2. Using PPO from Stable Baselines 3 (SB3)

- Configure the model and environment parameters in:  
  `tune/training/config/trimesh_config_PPO_SB3.yaml`

- Then run the following command from the `tune/` directory:
  ```bash
  python -m training.train_trimesh_SB3.py
  ```

###### Flip-Only Training (SB3 PPO)

To train an agent using only the flip action with SB3 PPO, run the training script in pycharm `tune/training/train_trimesh_flip_SB3.py`
> ❗ This environment may be deprecated.

---

### Quadrangular Meshes

For training on quadrangular meshes, you can use an agent with all four actions: **flip clockwise**, **flip counterclockwise**, **split**, and **collapse**. Two training models are available:

1. **Custom PPO Model** (`tune/model_RL/PPO_model_pers`)
2. **PPO from Stable Baselines 3 (SB3)**

#### 🚀 Starting Training

##### 1. Using `tune/model_RL/PPO_model`

- Configure the model and environment parameters in:  
  `tune/training/config/quadmesh_config_PPO_perso.yaml`

- Then run the following command from the `tune/` directory:
  ```bash
  python -m training.train_quadmesh.py
  ```
  
##### 2. Using PPO from Stable Baselines 3 (SB3)

- Configure the model and environment parameters in:  
  `tune/training/config/quadmesh_config_PPO_SB3.yaml`

- Then run the following command from the `tune/` directory:
  ```bash
  python -m training.train_quadmesh_SB3.py
  ```
  
#### 🧪 Testing a Saved SB3 Policy

After training, the model is saved as a `.zip` file in the `tune/training/policy_saved/` directory. To evaluate the policy, follow these steps in `tune/training/exploit_SB3_policy.py` :

##### 1. Create a Test Dataset

You can either:

- **Load a specific mesh file and duplicate it**:
  ```python
  mesh = read_gmsh("../mesh_files/t1_quad.msh")
  dataset = [mesh for _ in range(9)]
  ```

- **Generate a set of random quad meshes**:
  ```python
  dataset = [QM.random_mesh() for _ in range(9)]
  ```

##### 2. Load the Environment Configuration

Make sure to change and load the environment settings before testing:

```python
with open("../environment/environment_config.json", "r") as f:
    env_config = json.load(f)

plot_dataset(dataset)
```

##### 3. Load the Model

Use the `PPO.load()` function and evaluate the policy with your dataset:

```python
model = PPO.load("policy_saved/name.zip")
```

##### 4. Run the script

Run the script directly in **PyCharm** (or another IDE that supports graphical output) instead of the terminal.  
> ❗ If executed in a terminal without GUI support, the plots will not be displayed.


#### 🧪 Testing a Saved PPO_perso Policy

🚧 *Section in progress...*
