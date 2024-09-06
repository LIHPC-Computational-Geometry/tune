from typing import Any
import math
import numpy as np
from model.mesh_analysis import global_score, isValidAction, find_template_opposite_node
from model.mesh_struct.mesh_elements import Dart
from model.mesh_struct.mesh import Mesh
from actions.triangular_actions import flip_edge, split_edge, collapse_edge
from model.random_trimesh import random_flip_mesh, random_mesh

# possible actions
FLIP = 0
SPLIT = 1
COLLAPSE = 2
GLOBAL = 0


class TriMesh:
    def __init__(self, mesh=None, mesh_size: int = None, max_steps: int = 50, feat: int = 0):
        self.mesh = mesh if mesh is not None else random_flip_mesh(mesh_size)
        self.mesh_size = len(self.mesh.active_nodes())
        self.size = len(self.mesh.dart_info)
        self.actions = np.array([FLIP, SPLIT, COLLAPSE])
        self.reward = 0
        self.steps = 0
        self.max_steps = max_steps
        self.nodes_scores, self.mesh_score, self.ideal_score = global_score(self.mesh)
        self.terminal = False
        self.feat = feat
        self.won = 0

    def reset(self, mesh=None):
        self.reward = 0
        self.steps = 0
        self.terminal = False
        self.mesh = mesh if mesh is not None else random_mesh(self.mesh_size)
        self.size = len(self.mesh.dart_info)
        self.nodes_scores, self.mesh_score, self.ideal_score = global_score(self.mesh)
        self.won = 0

    def step(self, action):
        dart_id = action[1]
        d = Dart(self.mesh, dart_id)
        d1 = d.get_beta(1)
        n1 = d.get_node()
        n2 = d1.get_node()
        if action[2] == FLIP:
            flip_edge(self.mesh, n1, n2)
        elif action[2] == SPLIT:
            split_edge(self.mesh, n1, n2)
        elif action[2] == COLLAPSE:
            collapse_edge(self.mesh, n1, n2)
        self.steps += 1
        next_nodes_score, next_mesh_score, _ = global_score(self.mesh)
        self.nodes_scores = next_nodes_score
        self.reward = (self.mesh_score - next_mesh_score)*10
        if self.steps >= self.max_steps or next_mesh_score == self.ideal_score:
            if next_mesh_score == self.ideal_score:
                self.won = True
            self.terminal = True
        self.nodes_scores, self.mesh_score = next_nodes_score, next_mesh_score

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
    template = get_template_2(mesh)
    darts_to_delete = []
    darts_id = []
    all_action_type = 3

    for i, d_info in enumerate(mesh.active_darts()):
        d_id = d_info[0]
        d = Dart(mesh, d_id)
        if d_info[2] == -1: #test the validity of all action type
            darts_to_delete.append(i)
        else:
            darts_id.append(d_id)
    valid_template = np.delete(template, darts_to_delete, axis=0)
    score_sum = np.sum(np.abs(valid_template), axis=1)
    indices_top_10 = np.argsort(score_sum)[-5:][::-1]
    valid_dart_ids = [darts_id[i] for i in indices_top_10]
    X = valid_template[indices_top_10, :]
    X = X.flatten()
    return X, valid_dart_ids


def get_template_2(mesh: Mesh):
    nodes_scores = global_score(mesh)[0]
    size = len(mesh.active_darts())
    template = np.zeros((size, 6))

    for i, d_info in enumerate(mesh.active_darts()):

        d = Dart(mesh, d_info[0])
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()

        # Template niveau 1
        template[i, 0] = nodes_scores[C.id] if not math.isnan(nodes_scores[C.id]) else 0
        template[i, 1] = nodes_scores[A.id] if not math.isnan(nodes_scores[A.id]) else 0
        template[i, 2] = nodes_scores[B.id] if not math.isnan(nodes_scores[B.id]) else 0

        # template niveau 2

        n_id = find_template_opposite_node(d)
        if n_id is not None and not math.isnan(nodes_scores[n_id]):
            template[i, 3] = nodes_scores[n_id]
        n_id = find_template_opposite_node(d1)
        if n_id is not None and not math.isnan(nodes_scores[n_id]):
            template[i, 4] = nodes_scores[n_id]
        n_id = find_template_opposite_node(d11)
        if n_id is not None and not math.isnan(nodes_scores[n_id]):
            template[i, 5] = nodes_scores[n_id]

    return template
