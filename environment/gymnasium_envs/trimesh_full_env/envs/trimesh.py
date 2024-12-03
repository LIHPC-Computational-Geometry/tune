from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from mesh_model.random_trimesh import random_mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis import global_score
from environment.gymnasium_envs.trimesh_full_env.envs.mesh_conv import get_x_level_2
from actions.triangular_actions import flip_edge, split_edge, collapse_edge

from view.window import Game
from mesh_display import MeshDisplay



class Actions(Enum):
    FLIP = 0
    SPLIT = 1
    COLLAPSE = 2
    
class TriMeshEnvFull(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, mesh=None, mesh_size=16, render_mode=None, size=5):
        self.mesh = mesh if mesh is not None else random_mesh(mesh_size)
        self.mesh_size = len(self.mesh.nodes)
        self.nb_darts = len(self.mesh.dart_info)
        self._nodes_scores, self._mesh_score, self._ideal_score = global_score(self.mesh)
        self.deep = 6
        self.window_size = 512  # The size of the PyGame window
        self.g = None

        self.observation_space = spaces.Dict(
            {
                "irregularities": spaces.Box(-15, 5, shape=(30,), dtype=int), # nodes max degree : 15
                "darts_list": spaces.Box( 0, self.nb_darts*2, shape=(5,), dtype=int),
            }
        )
        self.observation = None
        # We have 3 action, flip, split, collapse
        self.n_darts_selected = 4 # actions restricted to 5 darts
        self.action_space = spaces.Box(low=0, high=np.array([3,self.n_darts_selected]), dtype=np.int64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def reset(self, seed=None, mesh=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Generation of a new irregular mesh
        self.mesh = mesh if mesh is not None else random_mesh(self.mesh_size)
        self.nb_darts = len(self.mesh.dart_info)
        self._nodes_scores, self._mesh_score, self._ideal_score = global_score(self.mesh)
        self.close()
        self.observation = self._get_obs()
        info = self._get_info(terminated=False)

        if self.render_mode == "human":
            self._render_frame()

        return self.observation, info


    def _get_obs(self):
        irregularities, darts_list = get_x_level_2(self.mesh)
        return {"irregularities": irregularities, "darts_list": darts_list}

    def _get_info(self, terminated):
        return {
            "distance": self._mesh_score - self._ideal_score,
            "is_success": 1.0 if terminated else 0.0,
        }

    def _action_to_dart_id(self, action):
        obs = self.observation
        darts_list = obs["darts_list"]
        return darts_list[int(action[1])]

    def step(self, action):
        dart_id_to_flip = self._action_to_dart_id(action)
        d = Dart(self.mesh, dart_id_to_flip)
        d1 = d.get_beta(1)
        n1 = d.get_node()
        n2 = d1.get_node()
        if action[0] == Actions.FLIP.value:
            flip_edge(self.mesh, n1, n2)
        elif action[0] == Actions.SPLIT.value:
            split_edge(self.mesh, n1, n2)
        elif action[0] == Actions.COLLAPSE.value:
            collapse_edge(self.mesh, n1, n2)
            
        # An episode is done if the actual score is the same as the ideal
        next_nodes_score, next_mesh_score, _ = global_score(self.mesh)
        terminated = np.array_equal(self._ideal_score, next_mesh_score)
        reward = (self._mesh_score - next_mesh_score)*10
        self._nodes_scores, self._mesh_score = next_nodes_score, next_mesh_score
        self.observation = self._get_obs()
        info = self._get_info(terminated)

        if self.render_mode == "human":
            self._render_frame()

        return self.observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human": #ou rgb par défaut
            return self._render_frame()

    def _render_frame(self):
        if self.g is None and self.render_mode == "human":
            mesh_disp = MeshDisplay(self.mesh)
            self.g = Game(self.mesh, mesh_disp)
            self.window = True
            self.g.control_events()
            self.g.window.fill((255, 255, 255))
            self.g.draw()
            pygame.display.flip()
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.render_mode == "human":
            self.g.mesh_disp = MeshDisplay(self.mesh)
            self.g.control_events()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.g = None
