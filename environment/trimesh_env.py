import numpy as np
import features.conv as FT
from model.mesh_analysis import global_score
from model.mesh_struct.mesh_elements import Dart
from actions.triangular_actions import flip_edge, flip_edge_ids
from model.random_trimesh import regular_mesh, random_flip_mesh

# possible actions
FLIP = 0


class TriMesh:
    def __init__(self, mesh_size, feat):
        self.mesh = random_flip_mesh(mesh_size)
        self.size = len(self.mesh.dart_info)
        self.actions = np.array([FLIP])
        self.reward = 0
        self.steps = 0
        self.nodes_scores = global_score(self.mesh)[0]
        self.ideal_score = global_score(self.mesh)[2]
        self.terminal = False
        self.feat = 0
        self.won = 0

    def reset(self):
        self.reward = 0
        self.steps = 0
        self.terminal = False
        self.mesh = random_flip_mesh(12)
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
        if self.steps >= 50 or next_mesh_score == mesh_ideal_score:
            if next_mesh_score == mesh_ideal_score:
                self.won = True
            self.terminal = True

    def get_x(self, s, a):
        if s is None:
            s = self.mesh
        X = FT.get_x(s, a, self, self.feat)
        return X
