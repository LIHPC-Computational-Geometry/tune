from mesh_model.random_quadmesh import random_mesh
from environment.gymnasium_envs import quadmesh_env
from mesh_model.reader import read_gmsh

from view.mesh_plotter.create_plots import plot_training_results, plot_test_results
from view.mesh_plotter.mesh_plots import plot_dataset

from model_RL.evaluate_model import testPolicy

from model_RL.PPO_model_pers import PPO

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import random
import torch
import numpy as np
import time
import wandb
import json
import os

if __name__ == '__main__':

    # SEEDING
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    with open("model_RL/parameters/ppo_config.json", "r") as f:
        ppo_config = json.load(f)
    with open("environment/environment_config.json", "r") as f:
        env_config = json.load(f)

    # Create log dir
    log_dir = ppo_config["tensorboard_log"]
    os.makedirs(log_dir, exist_ok=True)

    # Create the environment
    env = gym.make(
        env_config["env_name"],
        mesh=read_gmsh("mesh_files/simple_quad.msh"),
        max_episode_steps=env_config["max_episode_steps"],
        n_darts_selected=env_config["n_darts_selected"],
        deep=env_config["deep"],
        action_restriction=env_config["action_restriction"],
        with_degree_obs=env_config["with_degree_observation"]
    )

    model = PPO(
        env=env,
        lr=ppo_config["learning_rate"],
        gamma=ppo_config["gamma"],
        nb_iterations=20,
        nb_episodes_per_iteration=100,
        nb_epochs=5,
        batch_size=8
    )

    run_name = f"{env_config['env_name']}__{1}__{int(time.time())}"
    # Create log dir
    log_dir = ppo_config["tensorboard_log"]
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(f"results/runs/{run_name}")
    writer.add_text(
        "Environment config",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in env_config.items()])),
    )
    writer.add_text(
        "PPO config",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in ppo_config.items()])),
    )

    actor, rewards, wins, steps = model.learn(writer)
    writer.close()
    if rewards is not None:
        plot_training_results(rewards, wins, steps)
    # torch.save(actor.state_dict(), 'policy_saved/actor_network.pth')
