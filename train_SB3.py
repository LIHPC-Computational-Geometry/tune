from __future__ import annotations

import os
from math import gamma

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


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._plot = None

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if self._plot is None:  # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else:  # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02,
                                     self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True, True, True)
            self._plot[-1].canvas.draw()
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)

            self.logger.record("episode_reward", self.current_episode_reward)
            self.logger.record("episode_length", self.current_episode_length)

            is_success = self.locals["infos"][0].get("is_success", 0.0)  # Valeur par défaut : 0.0
            self.logger.record("episode_success", is_success)

            self.logger.dump(step=self.episode_count)
            self.current_episode_reward = 0  # Réinitialise la récompense de l'épisode
            self.current_episode_length = 0
            is_success = 0.0
            self.episode_count += 1  # Incrémente le compteur d'épisodes

        # Log scalar value (here a random variable)
        # value = np.random.random()
        # self.logger.record("random_value", value)
        return True


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
#env = make_vec_env('TrimeshFull-v0', n_envs=1, monitor_dir=log_dir)
env = gym.make("TrimeshFull-v0", max_episode_steps=100)
check_env(env, warn=True)

plotting_callback = PlottingCallback()

model = PPO("MultiInputPolicy", env, n_steps=1000, n_epochs=5, batch_size=8, learning_rate=1e-4, gamma=0.9,  verbose=1, tensorboard_log="/tmp/gym/") #./trimesh_tensorboard/

print("-----------Starting learning-----------")
model.learn(total_timesteps=10000, callback=TensorboardCallback())
print("-----------Learning ended------------")
model.save("ppo_trimesh_v0")
"""

model = PPO.load("ppo_trimesh_v0")

env = gym.make("trimesh-v0", max_episode_steps=100, render_mode="human")
m = PPO("MultiInputPolicy", env, verbose=1)
vec_env = m.get_env()
obs = vec_env.reset()
avg_rewards = []
ep_rewards = 0
nb_episodes = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    time.sleep(1)
    ep_rewards += reward
    if done:
        avg_rewards.append(ep_rewards)
        ep_rewards = 0
        nb_episodes += 1
        obs = vec_env.reset()
        #vec_env.close()
"""
"""
plt.figure()
plt.plot(avg_rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Learning Rewards, nb_episodes={}'.format(nb_episodes))
plt.legend(loc="best")
plt.show()"""