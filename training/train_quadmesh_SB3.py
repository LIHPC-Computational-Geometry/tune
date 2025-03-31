from __future__ import annotations

import os
import json

import mesh_model.random_quadmesh as QM
from mesh_model.reader import read_gmsh
from view.mesh_plotter.mesh_plots import dataset_plt
from training.exploit_SB3_policy import testPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

from environment.gymnasium_envs import quadmesh_env

import gymnasium as gym

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, model, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.episode_rewards = []
        self.mesh_reward = 0
        self.current_episode_reward = 0
        self.episode_count = 0
        self.current_episode_length = 0
        self.actions_info = {
            "episode_valid_actions": 0,
            "episode_invalid_topo": 0,
            "episode_invalid_geo": 0,
            "nb_flip" : 0,
            "nb_split": 0,
            "nb_collapse": 0,
            "nb_cleanup": 0,
            "nb_invalid_flip": 0,
            "nb_invalid_split": 0,
            "nb_invalid_collapse": 0,
            "nb_invalid_cleanup": 0,
        }
        self.final_distance = 0
        self.normalized_return = 0

    def _on_training_start(self) -> None:
        """
        Record PPO parameters and environment configuration at the training start.
        """
        self.logger.record("parameters/ppo", f"<pre>{json.dumps(ppo_config, indent=4)}</pre>")
        self.logger.record("parameters/env", f"<pre>{json.dumps(env_config, indent=4)}</pre>")
        self.logger.dump(step=0)

    def _on_step(self) -> bool:
        """
        Record different learning variables to monitor
        """
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        self.actions_info["episode_valid_actions"] += self.locals["infos"][0].get("valid_action", 0.0)
        self.actions_info["episode_invalid_topo"] += self.locals["infos"][0].get("invalid_topo", 0.0)
        self.actions_info["episode_invalid_geo"] += self.locals["infos"][0].get("invalid_geo", 0.0)
        self.actions_info["nb_flip"] += self.locals["infos"][0].get("flip", 0.0)
        self.actions_info["nb_split"] += self.locals["infos"][0].get("split", 0.0)
        self.actions_info["nb_collapse"] += self.locals["infos"][0].get("collapse", 0.0)
        self.actions_info["nb_cleanup"] += self.locals["infos"][0].get("cleanup", 0.0)
        self.actions_info["nb_invalid_flip"] += self.locals["infos"][0].get("invalid_flip", 0.0)
        self.actions_info["nb_invalid_split"] += self.locals["infos"][0].get("invalid_split", 0.0)
        self.actions_info["nb_invalid_collapse"] += self.locals["infos"][0].get("invalid_collapse", 0.0)
        self.actions_info["nb_invalid_cleanup"] += self.locals["infos"][0].get("invalid_cleanup", 0.0)

        self.mesh_reward += self.locals["infos"][0].get("mesh_reward", 0.0)

        # When the episode is over
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward) # global rewards obtained during the episode
            mesh_ideal_reward = self.locals["infos"][0].get("mesh_ideal_rewards", 0.0) # maximum achievable reward
            if mesh_ideal_reward > 0:
                self.normalized_return = self.mesh_reward/ mesh_ideal_reward
            else:
                self.normalized_return = 0

            self.final_distance = self.locals["infos"][0].get("distance", 0.0)
            self.logger.record("final_distance", self.final_distance)
            self.logger.record("valid_actions", self.actions_info["episode_valid_actions"]*100/self.current_episode_length if self.current_episode_length > 0 else 0)
            self.logger.record("n_invalid_topo", self.actions_info["episode_invalid_topo"])
            self.logger.record("n_invalid_geo", self.actions_info["episode_invalid_geo"])
            self.logger.record("nb_flip", self.actions_info["nb_flip"])
            self.logger.record("nb_split", self.actions_info["nb_split"])
            self.logger.record("nb_collapse", self.actions_info["nb_collapse"])
            self.logger.record("nb_cleanup", self.actions_info["nb_cleanup"])
            self.logger.record("invalid_flip", self.actions_info["nb_invalid_flip"]*100/self.actions_info["nb_flip"] if self.actions_info["nb_flip"] > 0 else 0)
            self.logger.record("invalid_split", self.actions_info["nb_invalid_split"]*100/self.actions_info["nb_split"] if self.actions_info["nb_split"] > 0 else 0)
            self.logger.record("invalid_collapse", self.actions_info["nb_invalid_collapse"]*100/self.actions_info["nb_collapse"]if self.actions_info["nb_collapse"] > 0 else 0)
            self.logger.record("invalid_cleanup", self.actions_info["nb_invalid_cleanup"]*100/self.actions_info["nb_cleanup"]if self.actions_info["nb_cleanup"] > 0 else 0)
            self.logger.record("episode_mesh_reward", self.mesh_reward)
            self.logger.record("episode_reward", self.current_episode_reward)
            self.logger.record("normalized_return", self.normalized_return)
            self.logger.record("episode_length", self.current_episode_length)

            is_success = self.locals["infos"][0].get("is_success", 0.0)  # Default value: 0.0
            self.logger.record("episode_success", is_success)

            self.logger.dump(step=self.episode_count)
            self.current_episode_reward = 0  #  resets global episode reward
            self.mesh_reward = 0 #  resets mesh episode reward
            self.current_episode_length = 0
            #reset actions info
            for key in self.actions_info.keys():
                self.actions_info[key] = 0
            self.episode_count += 1  # Increment episode counter

        return True

    def _on_training_end(self) -> None:
        """
        Records policy evaluation results : before and after dataset images
        """
        filename = "counts.json"
        counts_registry = self.locals["infos"][0].get("observation_count", 0.0)
        counts = counts_registry.counts

        # Convertir les clés tuple en chaînes de caractères
        counts_str_keys = {v: str(k) for k, v in counts.items()}

        # Écriture dans un fichier JSON
        with open(filename, "w") as file:
            json.dump(counts_str_keys, file, indent=4)

        print(f"Counts saved at {filename}")

        dataset = [QM.random_mesh() for _ in range(9)] # dataset of 9 meshes of size 30
        before = dataset_plt(dataset) # plot the datasat as image
        length, wins, rewards, normalized_return, final_meshes = testPolicy(self.model, 10, env_config, dataset) # test model policy on the dataset
        after = dataset_plt(final_meshes)
        self.logger.record("figures/before", Figure(before, close=True), exclude=("stdout", "log"))
        self.logger.record("figures/after", Figure(after, close=True), exclude=("stdout", "log"))
        self.logger.dump(step=0)


with open("../model_RL/parameters/ppo_config.json", "r") as f:
    ppo_config = json.load(f)
with open("../environment/environment_config.json", "r") as f:
    env_config = json.load(f)

# Create log dir
log_dir = ppo_config["tensorboard_log"]
os.makedirs(log_dir, exist_ok=True)

# Create the environment
env = gym.make(
    env_config["env_name"],
    mesh = read_gmsh("../mesh_files/exemple.msh"),
    max_episode_steps=env_config["max_episode_steps"],
    n_darts_selected=env_config["n_darts_selected"],
    deep= env_config["deep"],
    action_restriction=env_config["action_restriction"],
    with_degree_obs=env_config["with_degree_observation"]
)

check_env(env, warn=True)

model = PPO(
    policy=ppo_config["policy"],
    env=env,
    n_steps=ppo_config["n_steps"],
    n_epochs=ppo_config["n_epochs"],
    batch_size=ppo_config["batch_size"],
    learning_rate=ppo_config["learning_rate"],
    gamma=ppo_config["gamma"],
    verbose=ppo_config["verbose"],
    tensorboard_log=log_dir
)

print("-----------Starting learning-----------")
model.learn(total_timesteps=ppo_config["total_timesteps"], callback=TensorboardCallback(model))
print("-----------Learning ended------------")
model.save("policy_saved/quad/test3")