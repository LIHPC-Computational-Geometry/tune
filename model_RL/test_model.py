from environment.trimesh_env import TriMesh
from model_RL.PPO_model import PPO
from model_RL.actor_critic_networks import Actor, Critic
import numpy as np
import copy
import torch
import random
from tqdm import tqdm
from model_RL.actor_critic_networks import NaNExceptionActor, NaNExceptionCritic


def testPolicy(policy, n_eval_episodes, dataset):
    avg_length = np.zeros(len(dataset))
    avg_rewards = np.zeros(len(dataset))
    nb_wins = np.zeros(len(dataset))
    for i, mesh in enumerate(dataset, 0):
        env = TriMesh(mesh)
        for ep in range(n_eval_episodes):
            ep_rewards: int = 0
            ep_length: int = 0
            env.reset(mesh)
            while env.won == 0 and ep_length < 30:
                action = policy.select_action(env.mesh)
                env.step(action)
                ep_rewards += env.reward
                ep_length += 1
            if env.won == 1:
                nb_wins[i] += 1
            avg_length[i] += ep_length
            avg_rewards[i] += ep_rewards
        avg_length[i] = avg_length[i]/n_eval_episodes
        avg_rewards[i] = avg_rewards[i]/n_eval_episodes
    return avg_length, nb_wins, avg_rewards