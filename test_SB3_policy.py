from numpy import ndarray

import gymnasium as gym

from environment.gymnasium_envs.trimesh_full_env import TriMeshEnvFull
from stable_baselines3 import PPO
from mesh_model.mesh_analysis import global_score
from mesh_model.mesh_struct.mesh import Mesh
import mesh_model.random_trimesh as TM
from plots.create_plots import plot_test_results
from plots.mesh_plotter import plot_dataset
import numpy as np
import copy
from tqdm import tqdm


def testPolicy(
        model,
        n_eval_episodes: int,
        dataset: list[Mesh]
) -> tuple[ndarray, ndarray, ndarray, list[Mesh]]:
    """
    Tests policy on each mesh of a dataset with n_eval_episodes.
    :param policy: the policy to test
    :param n_eval_episodes: number of evaluation episodes on each mesh
    :param dataset: list of mesh objects
    :param max_steps: max steps to evaluate
    :return: average length of evaluation episodes, number of wins,average reward per mesh, dataset with the modified meshes
    """
    print('Testing policy')
    avg_length = np.zeros(len(dataset))
    avg_rewards = np.zeros(len(dataset))
    nb_wins = np.zeros(len(dataset))
    final_meshes = []
    for i, mesh in tqdm(enumerate(dataset, 1)):
        best_mesh = None
        env = gym.make("TrimeshFull-v0", mesh=mesh, max_episode_steps=200)
        for _ in range(n_eval_episodes):
            terminated = False
            truncated = False
            ep_rewards: int = 0
            ep_length: int = 0
            obs, info = env.reset(options={"mesh": mesh})
            while terminated == False and truncated == False:
                action, _states = model.predict(obs, deterministic=True)
                if action is None:
                    env.terminal = True
                    break
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rewards += reward
                ep_length += 1
            if terminated:
                nb_wins[i-1] += 1
            if isBetterMesh(best_mesh, info['mesh']):
                best_mesh = copy.deepcopy(info['mesh'])
            avg_length[i-1] += ep_length
            avg_rewards[i-1] += ep_rewards
        final_meshes.append(best_mesh)
        avg_length[i-1] = avg_length[i-1]/n_eval_episodes
        avg_rewards[i-1] = avg_rewards[i-1]/n_eval_episodes
    return avg_length, nb_wins, avg_rewards, final_meshes


def isBetterPolicy(actual_best_policy, policy_to_test):
    if actual_best_policy is None:
        return True

def isBetterMesh(best_mesh, actual_mesh):
    if best_mesh is None or global_score(best_mesh)[1] > global_score(actual_mesh)[1]:
        return True
    else:
        return False


dataset = [TM.random_mesh(30) for _ in range(9)]
plot_dataset(dataset)
model = PPO.load("ppo_trimesh_v5_p2")
avg_steps, avg_wins, avg_rewards, final_meshes = testPolicy(model, 10, dataset)

plot_test_results(avg_rewards, avg_wins, avg_steps)
plot_dataset(final_meshes)
