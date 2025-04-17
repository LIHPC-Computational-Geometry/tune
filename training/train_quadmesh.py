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
from datetime import datetime
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

    # Create the environment
    env = gym.make(
        env_config["env_name"],
        mesh=read_gmsh("mesh_files/medium_quad.msh"),
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
        nb_iterations=250,
        nb_episodes_per_iteration=100,
        nb_epochs=10,
        batch_size=16
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{env_config['env_name']}__medium_fix_batch16__{timestamp}"
    # Create log dir
    log_dir = ppo_config["tensorboard_log"]
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(f"training/results/quad_perso/{run_name}")
    writer.add_text(
        "Environment config",
        " | param | value | \n | - | - | \n%s" % ("\n".join([f" | {key} | {value} | " for key, value in env_config.items()])),
    )
    writer.add_text(
        "PPO config",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in ppo_config.items()])),
    )

    actor, rewards, wins, steps, counts_registry = model.learn(writer)
    writer.close()

    filename = "counts_perso_PPO51-medium.json"
    counts = counts_registry.counts

    # Convertir les clés tuple en chaînes de caractères
    counts_str_keys = {v: str(k) for k, v in counts.items()}

    # Écriture dans un fichier JSON
    with open(filename, "w") as file:
        json.dump(counts_str_keys, file, indent=4)

    print(f"Counts saved at {filename}")

    torch.save(actor.state_dict(), 'training/policy_saved/quad/PPO51-medium-perso.pth')
