from typing import Any
import numpy as np
from model.mesh_analysis import global_score, isValidAction, find_template_opposite_node
from model.mesh_struct.mesh_elements import Dart
from model.mesh_struct.mesh import Mesh
from actions.triangular_actions import flip_edge
from model.random_trimesh import random_flip_mesh

# possible actions
FLIP = 0
GLOBAL = 0


class TriMesh:
    def __init__(self, mesh=None, mesh_size: int = None, max_steps: int = 50, feat: int = 0):
        self.mesh = mesh if mesh is not None else random_flip_mesh(mesh_size)
        self.mesh_size = len(self.mesh.nodes)
        self.size = len(self.mesh.dart_info)
        self.actions = np.array([FLIP])
        self.reward = 0
        self.steps = 0
        self.max_steps = max_steps
        self.nodes_scores = global_score(self.mesh)[0]
        self.ideal_score = global_score(self.mesh)[2]
        self.terminal = False
        self.feat = feat
        self.won = 0

    def reset(self, mesh=None):
        self.reward = 0
        self.steps = 0
        self.terminal = False
        self.mesh = mesh if mesh is not None else random_flip_mesh(self.mesh_size)
        self.size = len(self.mesh.dart_info)
        self.nodes_scores = global_score(self.mesh)[0]
        self.ideal_score = global_score(self.mesh)[2]
        self.won = 0

    def step(self, action):
        dart_id = action[1]
        nodes_score, mesh_score, mesh_ideal_score = global_score(self.mesh)
        d = Dart(self.mesh, dart_id)
        d1 = d.get_beta(1)
        n1 = d.get_node()
        n2 = d1.get_node()
        flip_edge(self.mesh, n1, n2)
        self.steps += 1
        next_nodes_score, next_mesh_score, next_mesh_ideal_score = global_score(self.mesh)
        self.nodes_scores = next_nodes_score
        self.reward = (mesh_score - next_mesh_score)*10
        if self.steps >= self.max_steps or next_mesh_score == mesh_ideal_score:
            if next_mesh_score == mesh_ideal_score:
                self.won = True
            self.terminal = True

    def get_x(self, s: Mesh, a: int) -> tuple[Any, list[int | list[int]]]:
        """
        Get the feature vector of the state-action pair
        :param s: the state
        :param a: the action
        :return: the feature vector and valid darts id
        """
        if s is None:
            s = self.mesh
        if self.feat == GLOBAL:
            return get_x_global_4(self, s)


def get_x_global_4(env, state: Mesh) -> tuple[Any, list[int | list[int]]]:
    """
    Get the feature vector of the state.
    :param state: the state
    :param env: The environment
    :return: the feature vector
    """
    mesh = state
    nodes_scores = global_score(mesh)[0]
    size = len(mesh.dart_info)
    template = np.zeros((size, 6))

    for d_info in mesh.dart_info:

        d = Dart(mesh, d_info[0])
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()

        #Template niveau 1
        template[d_info[0], 0] = nodes_scores[C.id]
        template[d_info[0], 1] = nodes_scores[A.id]
        template[d_info[0], 2] = nodes_scores[B.id]

        #template niveau 2

        n_id = find_template_opposite_node(d)
        if n_id is not None:
            template[d_info[0], 3] = nodes_scores[n_id]
        n_id = find_template_opposite_node(d1)
        if n_id is not None:
            template[d_info[0], 4] = nodes_scores[n_id]
        n_id = find_template_opposite_node(d11)
        if n_id is not None:
            template[d_info[0], 5] = nodes_scores[n_id]

    dart_to_delete = []
    dart_ids = []
    for i in range(size):
        d = Dart(mesh, i)
        if not isValidAction(mesh, d.id):
            dart_to_delete.append(i)
        else :
            dart_ids.append(i)
    valid_template = np.delete(template, dart_to_delete, axis=0)
    score_sum = np.sum(np.abs(valid_template), axis=1)
    indices_top_10 = np.argsort(score_sum)[-5:][::-1]
    valid_dart_ids = [dart_ids[i] for i in indices_top_10]
    X = valid_template[indices_top_10, :]
    X = X.flatten()
    return X, valid_dart_ids
