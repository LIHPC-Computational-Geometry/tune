from __future__ import annotations

import os
import json

import mesh_model.random_trimesh as TM
from view.mesh_plotter.mesh_plots import dataset_plt
from training.exploit_SB3_policy import testPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from environment.gymnasium_envs import trimesh_flip_env

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
        }
        self.final_distance = 0
        self.normalized_return = 0

    def _on_training_start(self) -> None:
        self.logger.record("parameters/ppo", f"<pre>{json.dumps(ppo_config, indent=4)}</pre>")
        self.logger.record("parameters/env", f"<pre>{json.dumps(env_config, indent=4)}</pre>")
        self.logger.dump(step=0)

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        self.actions_info["episode_valid_actions"] += self.locals["infos"][0].get("valid_action", 0.0)
        self.actions_info["episode_invalid_topo"] += self.locals["infos"][0].get("invalid_topo", 0.0)
        self.actions_info["episode_invalid_geo"] += self.locals["infos"][0].get("invalid_geo", 0.0)

        self.mesh_reward += self.locals["infos"][0].get("mesh_reward", 0.0)

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            mesh_ideal_reward = self.locals["infos"][0].get("mesh_ideal_rewards", 0.0)
            if mesh_ideal_reward > 0:
                self.normalized_return = self.mesh_reward/ mesh_ideal_reward
            else:
                self.normalized_return = 0
            self.final_distance = self.locals["infos"][0].get("distance", 0.0)
            self.logger.record("final_distance", self.final_distance)
            self.logger.record("valid_actions", self.actions_info["episode_valid_actions"]*100/self.current_episode_length if self.current_episode_length > 0 else 0)
            self.logger.record("n_invalid_topo", self.actions_info["episode_invalid_topo"])
            self.logger.record("n_invalid_geo", self.actions_info["episode_invalid_geo"])

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
        Records policy evaluation trimesh_results : before and after dataset images
        """
        dataset = [TM.random_mesh(30) for _ in range(9)]  # dataset of 9 meshes of size 30
        before = dataset_plt(dataset)  # plot the datasat as image
        length, wins, rewards, normalized_return, final_meshes = testPolicy(self.model, 10, env_config, dataset)  # test model policy on the dataset
        after = dataset_plt(final_meshes)
        self.logger.record("figures/before", Figure(before, close=True), exclude=("stdout", "log"))
        self.logger.record("figures/after", Figure(after, close=True), exclude=("stdout", "log"))
        self.logger.dump(step=0)


with open("../model_RL/old_files/ppo_config.json", "r") as f:
    ppo_config = json.load(f)
with open("../environment/old_files/environment_config.json", "r") as f:
    env_config = json.load(f)

# Create log dir
log_dir = ppo_config["tensorboard_log"]
os.makedirs(log_dir, exist_ok=True)

# Create the environment
env = gym.make(
    env_config["env_name"],
    mesh_size=env_config["mesh_size"],
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
model.save("flip")