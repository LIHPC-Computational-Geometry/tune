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
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
import json
import os

if __name__ == '__main__':

    experiment_name = "random_basic_quad_perso-5"
    description = "Sans degrés, pénalité -5 et -1. deep = 12"
    ppo_config_path = "model_RL/parameters/ppo_config.json"
    env_config_path = "environment/environment_config.json"
    eval_env_config_path = "environment/eval_environment_config.json"
    policy_saving_path = os.path.join("training/policy_saved/quad-perso/", experiment_name)

    wandb_model_save_path = f"training/wandb_models/{experiment_name}"
    wandb.init(
        project="Quadmesh-learning",
        sync_tensorboard=True,
        save_code=True
    )

    # Mesh datasets
    evaluation_mesh_file_path = "mesh_files/simple_quad.msh"
    training_mesh_file_path = "mesh_files/simple_quad.msh"

    # SEEDING
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # PARAMETERS CONFIGURATION

    with open(ppo_config_path, "r") as f:
        ppo_config = json.load(f)
    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    with open(eval_env_config_path, "r") as f:
        eval_env_config = json.load(f)

    # Separate evaluation env
    eval_env = gym.make(
        eval_env_config["env_name"],
        #mesh=read_gmsh(evaluation_mesh_file_path),
        max_episode_steps=eval_env_config["max_episode_steps"],
        n_darts_selected=eval_env_config["n_darts_selected"],
        deep=eval_env_config["deep"],
        action_restriction=eval_env_config["action_restriction"],
        with_degree_obs=eval_env_config["with_degree_observation"]
    )

    # Create the environment
    env = gym.make(
        env_config["env_name"],
        #mesh = read_gmsh(training_mesh_file_path),
        max_episode_steps=env_config["max_episode_steps"],
        n_darts_selected=env_config["n_darts_selected"],
        deep=env_config["deep"],
        action_restriction=env_config["action_restriction"],
        with_degree_obs=env_config["with_degree_observation"]
    )

    model = PPO(
        env=env,
        obs_size= env_config["n_darts_selected"]*env_config["deep"]*2 if env_config["with_degree_observation"] else env_config["n_darts_selected"]*env_config["deep"],
        max_steps=env_config["max_episode_steps"],
        lr=ppo_config["learning_rate"],
        gamma=ppo_config["gamma"],
        nb_iterations=150,
        nb_episodes_per_iteration=100,
        nb_epochs=10,
        batch_size=16
    )

    # Create log dir
    log_dir = ppo_config["tensorboard_log"]
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(f"training/results/quad_perso/{experiment_name}")
    writer.add_text(
        "Environment config",
        " | param | value | \n | - | - | \n%s" % ("\n".join([f" | {key} | {value} | " for key, value in env_config.items()])),
    )
    writer.add_text(
        "PPO config",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in ppo_config.items()])),
    )
    writer.add_text("Description", description)

    actor, rewards, wins, steps, counts_registry = model.learn(writer)
    writer.close()

    filename = "counts_" + experiment_name + ".json"
    counts = counts_registry.counts

    # Convertir les clés tuple en chaînes de caractères
    counts_str_keys = [(v, str(k)) for k, v in counts.items()]
    counts_values = list(counts.values())

    # Écriture dans un fichier JSON
    with open(filename, "w") as file:
        json.dump(counts_str_keys, file, indent=4)

    print(f"Counts saved at {filename}")
    counts_values.sort()
    figure, ax = plt.subplots()
    ax.hist(counts_values, bins='auto')
    ax.set_title("Observation counts")
    writer.add_figure("observation counts", figure)
    writer.add_histogram("observation/counts", np.array(counts_values), bins='auto')

    torch.save(actor.state_dict(), policy_saving_path)

    wandb.finish()