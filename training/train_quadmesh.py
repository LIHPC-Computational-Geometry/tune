from mesh_model.random_quadmesh import random_mesh
from environment.gymnasium_envs import quadmesh_env
from mesh_model.reader import read_gmsh

from view.mesh_plotter.create_plots import plot_training_results, plot_test_results
from view.mesh_plotter.mesh_plots import plot_dataset

from model_RL.evaluate_model import testPolicy

from model_RL.PPO_model_pers import PPO

import gymnasium as gym
import json


def train():
    mesh_size = 30
    lr = 0.0001
    gamma = 0.9

    #dataset = [random_mesh() for _ in range(9)]
    #plot_dataset(dataset)

    with open("environment/environment_config.json", "r") as f:
        env_config = json.load(f)

    # Create the environment
    env = gym.make(
        env_config["env_name"],
        mesh=read_gmsh("mesh_files/exemple.msh"),
        max_episode_steps=env_config["max_episode_steps"],
        n_darts_selected=env_config["n_darts_selected"],
        deep=env_config["deep"],
        action_restriction=env_config["action_restriction"],
        with_degree_obs=env_config["with_degree_observation"]
    )

    model = PPO(env, lr, gamma, nb_iterations=15, nb_episodes_per_iteration=100, nb_epochs=5, batch_size=8)
    actor, rewards, wins, steps = model.learn()
    if rewards is not None:
        plot_training_results(rewards, wins, steps)

"""
    # torch.save(actor.state_dict(), 'policy_saved/actor_network.pth')
    avg_steps, avg_wins, avg_rewards, final_meshes = testPolicy(actor, 5, dataset, 60)

    if rewards is not None:
        plot_test_results(avg_rewards, avg_wins, avg_steps, avg_rewards)
    plot_dataset(final_meshes)
"""