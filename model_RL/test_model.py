from numpy import ndarray

from environment.trimesh_env import TriMesh
from model.mesh_analysis import global_score
from model.mesh_struct.mesh import Mesh
import numpy as np
import copy
from tqdm import tqdm


def testPolicy(
        policy,
        n_eval_episodes: int,
        dataset: list[Mesh],
        max_steps: int
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
        env = TriMesh(mesh, None, max_steps)
        best_mesh = None
        for _ in range(n_eval_episodes):
            ep_rewards: int = 0
            ep_length: int = 0
            env.reset(mesh)
            while env.won == 0 and ep_length < 30:
                action = policy.select_action(env.mesh)
                env.step(action)
                ep_rewards += env.reward
                ep_length += 1
            if env.won == 1:
                nb_wins[i-1] += 1
            if isBetterMesh(best_mesh, env.mesh):
                best_mesh = copy.deepcopy(env.mesh)
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
    if actual_mesh is not None or global_score(best_mesh)[1] > global_score(actual_mesh)[1]:
        return True
    else:
        return False
