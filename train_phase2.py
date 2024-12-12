from __future__ import annotations

import os
import json
from math import gamma
from typing import Dict, Any

from environment.gymnasium_envs.trimesh_env import TriMeshEnv
from environment.gymnasium_envs.trimesh_full_env import TriMeshEnvFull

from stable_baselines3 import PPO,SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

import numpy as np
import matplotlib.pyplot as plt
import time
import gymnasium as gym

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, model, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        self.current_episode_length = 0
        self.episode_valid_actions = 0
        self.episode_invalid_topo = 0
        self.episode_invalid_geo = 0
        self.final_distance = 0

    def _on_training_start(self) -> None:
        with open("model_RL/parameters/ppo_config_200k.json", "r") as f:
            ppo_config = json.load(f)
        self.logger.record("parameters/ppo", f"<pre>{json.dumps(ppo_config, indent=4)}</pre>")
        self.logger.dump(step=0)


    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        self.episode_valid_actions += self.locals["infos"][0].get("valid_action", 0.0)
        self.episode_invalid_topo += self.locals["infos"][0].get("invalid_topo", 0.0)
        self.episode_invalid_geo += self.locals["infos"][0].get("invalid_geo", 0.0)
        self.distance = self.locals["infos"][0].get("distance", 0.0)
        self.logger.record("distance", self.distance)

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.final_distance = self.distance
            self.logger.record("final_distance", self.final_distance)
            self.logger.record("n_valid_actions", self.episode_valid_actions)
            self.logger.record("n_invalid_topo", self.episode_invalid_topo)
            self.logger.record("n_invalid_geo", self.episode_invalid_geo)
            self.logger.record("episode_reward", self.current_episode_reward)
            self.logger.record("episode_length", self.current_episode_length)

            is_success = self.locals["infos"][0].get("is_success", 0.0)  # Valeur par défaut : 0.0
            self.logger.record("episode_success", is_success)

            self.logger.dump(step=self.episode_count)
            self.current_episode_reward = 0  # Réinitialise la récompense de l'épisode
            self.current_episode_length = 0
            self.episode_valid_actions = 0
            self.episode_invalid_geo = 0
            self.episode_invalid_topo = 0
            self.episode_count += 1  # Incrémente le compteur d'épisodes

        # Log scalar value (here a random variable)
        # value = np.random.random()
        # self.logger.record("random_value", value)
        return True


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

with open("model_RL/parameters/ppo_config_200k.json", "r") as f:
    ppo_config = json.load(f)

env = gym.make("TrimeshFull-v0", max_episode_steps=40)
check_env(env, warn=True)

model = PPO.load("ppo_trimesh_v5")
model.set_env(env)
print("-----------Starting learning-----------")
model.learn(total_timesteps=ppo_config["total_timesteps"], callback=TensorboardCallback(model))
print("-----------Learning ended------------")
model.save("ppo_trimesh_v5_p2")