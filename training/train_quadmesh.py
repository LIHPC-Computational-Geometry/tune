from __future__ import annotations

import random
import torch
import yaml
import os
import time
import wandb

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from torch.utils.tensorboard import SummaryWriter

#Internal import
from environment.gymnasium_envs import quadmesh_env
from mesh_model.reader import read_gmsh

from model_RL.PPO_model_pers import PPO


def log_init(log_writer, config):
    log_writer.add_text("Description", config["description"])
    log_writer.add_hparams(hparam_dict=config["env"], metric_dict=config["metrics"], run_name=config["experiment_name"])

def log_end(log_writer, config, obs_registry):
    if obs_registry is not None:
        filename = "training/observation_counts/counts_" + config["experiment_name"] + ".csv"

        obs_registry.save(filename)
        print(f"Counts saved at {filename}")

        counts = obs_registry.df["counts"]

        log_writer.add_scalar("observation/n_observation", len(counts))
        log_writer.add_scalar("observation/mean", np.mean(counts))
        log_writer.add_scalar("observation/median", np.median(counts))
        log_writer.add_scalar("observation/min", np.min(counts))
        log_writer.add_scalar("observation/max", np.max(counts))

        values = counts.values
        values = np.sort(values)[::-1]  # Descending sort
        max_values = 300
        if len(values) > max_values:
            values = values[:max_values]  # We keep only 500 values for histogram

        figure, ax = plt.subplots()
        ax.bar(range(len(values)), values)
        ax.set_title("Observation counts")

        log_writer.add_figure("observation/counts", figure)

if __name__ == '__main__':

    # PARAMETERS CONFIGURATION
    with open("training/config/quadmesh_config_PPO_perso.yaml", "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment_name"]

    # Create log dir
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    wandb.tensorboard.patch(root_logdir=log_dir)
    wandb.init(
        project="Quadmesh-learning",
        name=config["experiment_name"],
        config=config,
        sync_tensorboard=True,
        save_code=True
    )

    # SEEDING
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create the environment
    env = gym.make(
        config["env"]["env_id"],
        # mesh=read_gmsh(config["dataset"]["evaluation_mesh_file_path"]),
        max_episode_steps=config["env"]["max_episode_steps"],
        n_darts_selected=config["env"]["n_darts_selected"],
        deep=config["env"]["deep"],
        action_restriction=config["env"]["action_restriction"],
        with_degree_obs=config["env"]["with_degree_observation"],
        render_mode=config["env"]["render_mode"],
        obs_count=config["env"]["obs_count"],
    )

    model = PPO(
        env=env,
        obs_size= config["env"]["obs_size"],
        n_actions=config["ppo"]["n_actions"],
        n_darts_observed=config["env"]["n_darts_selected"],
        max_steps=config["env"]["max_episode_steps"],
        lr=config["ppo"]["learning_rate"],
        gamma=config["ppo"]["gamma"],
        nb_iterations=config["ppo"]["n_iterations"],
        nb_episodes_per_iteration=config["ppo"]["n_episodes_per_iteration"],
        nb_epochs=config["ppo"]["n_epochs"],
        batch_size=config["ppo"]["batch_size"],
    )

    writer = SummaryWriter(log_dir + config["experiment_name"])
    log_init(writer, config)

    # LEARNING
    start_time = time.perf_counter()
    print("-----------Starting learning-----------")

    actor, rewards, wins, steps, obs_registry = model.learn(writer)

    log_end(writer, config, obs_registry)

    end_time = time.perf_counter()
    print("-----------Learning ended------------")
    print(f"Temps d'apprentissage : {end_time - start_time:.4} s")

    # SAVING POLICY
    torch.save(actor.state_dict(), config["paths"]["policy_saving_dir"]+config["experiment_name"]+".pth")
    writer.close()
    wandb.finish()