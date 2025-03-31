from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

from mesh_model.random_quadmesh import random_mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis.global_mesh_analysis import global_score
from mesh_model.mesh_analysis.quadmesh_analysis import isTruncated
from environment.gymnasium_envs.quadmesh_env.envs.mesh_conv import get_x
from environment.actions.quadrangular_actions import flip_edge, split_edge, collapse_edge, cleanup_edge
from environment.observation_register import ObservationRegistry


class Actions(Enum):
    FLIP = 0
    SPLIT = 1
    COLLAPSE = 2
    CLEANUP = 3


class QuadMeshEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, mesh=None, max_episode_steps=30, n_darts_selected=20, deep=6, with_degree_obs=True, action_restriction=False, render_mode=None):
        if mesh is not None:
            self.config = {"mesh": mesh}
            self.mesh = copy.deepcopy(mesh)
        else :
            self.config = {"mesh": None}
            self.mesh = random_mesh()
        self.mesh_size = len(self.mesh.nodes)
        self.nb_darts = len(self.mesh.dart_info)
        self._nodes_scores, self._mesh_score, self._ideal_score, self._nodes_adjacency = global_score(self.mesh)
        self._ideal_rewards = (self._mesh_score - self._ideal_score)*10
        self.next_mesh_score = 0
        self.n_darts_selected = n_darts_selected
        self.restricted = action_restriction
        self.degree_observation = with_degree_obs
        self.window_size = 512  # The size of the PyGame window
        self.g = None
        self.nb_invalid_actions = 0
        self.max_steps = max_episode_steps
        self.darts_selected = [] # darts id observed
        self.deep = deep*2 if self.degree_observation else deep
        self.observation_space = spaces.Box(
            low=-6,  # nodes min degree : -15
            high=2,  # nodes max degree : 15
            shape=(self.n_darts_selected, deep),
            dtype=np.int64
        )
        self.observation_count = ObservationRegistry(self.n_darts_selected, self.deep, -6, 2)

        self.observation = None

        # We have 4 actions, flip, split, collapse, cleanup
        self.action_space = spaces.MultiDiscrete([3, self.n_darts_selected])


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if options is not None:
            self.mesh = options['mesh']
        elif self.config["mesh"] is not None:
            self.mesh = copy.deepcopy(self.config["mesh"])
        else:
            self.mesh = random_mesh()
        self.nb_darts = len(self.mesh.dart_info)
        self._nodes_scores, self._mesh_score, self._ideal_score, self._nodes_adjacency = global_score(self.mesh)
        self._ideal_rewards = (self._mesh_score - self._ideal_score) * 10
        self.nb_invalid_actions = 0
        self.close()
        self.observation = self._get_obs()
        info = self._get_info(terminated=False,valid_act=(None,None,None), action=(None,None), mesh_reward=None)

        if self.render_mode == "human":
            self._render_frame()

        return self.observation, info


    def _get_obs(self):
        irregularities, darts_list = get_x(self.mesh, self.n_darts_selected, self.deep, self.degree_observation, self.restricted, self._nodes_scores, self._nodes_adjacency)
        self.darts_selected = darts_list
        return irregularities

    def _get_info(self, terminated, valid_act, action, mesh_reward):
        valid_action, valid_topo, valid_geo = valid_act
        return {
            "distance": self._mesh_score - self._ideal_score,
            "mesh_reward" : mesh_reward,
            "mesh_ideal_rewards" : self._ideal_rewards,
            "is_success": 1.0 if terminated else 0.0,
            "valid_action": 1.0 if valid_action else 0.0,
            "invalid_topo": 1.0 if not valid_topo else 0.0,
            "invalid_geo": 1.0 if  not valid_geo else 0.0,
            "flip": 1.0 if action[0]==Actions.FLIP.value else 0.0,
            "split": 1.0 if action[0]==Actions.SPLIT.value else 0.0,
            "collapse": 1.0 if action[0]==Actions.COLLAPSE.value else 0.0,
            "cleanup": 1.0 if action[0]==Actions.CLEANUP.value else 0.0,
            "invalid_flip": 1.0 if action[0]==Actions.FLIP.value and not valid_action else 0.0,
            "invalid_split": 1.0 if action[0]==Actions.SPLIT.value and not valid_action else 0.0,
            "invalid_collapse": 1.0 if action[0]==Actions.COLLAPSE.value and not valid_action else 0.0,
            "invalid_cleanup": 1.0 if action[0]==Actions.CLEANUP.value and not valid_action else 0.0,
            "mesh" : self.mesh,
            "darts_selected" : self.darts_selected,
            "observation_count" : self.observation_count,
        }

    def _action_to_dart_id(self, action: np.ndarray) -> int:
        """
        Converts an action ID into the dart ID on which to perform the action
        :param action: action ID
        :return: the dart ID on which to perform the action
        """
        return self.darts_selected[int(action[1])]

    def step(self, action: np.ndarray):
        dart_id = self._action_to_dart_id(action)
        d = Dart(self.mesh, dart_id)
        d1 = d.get_beta(1)
        n1 = d.get_node()
        n2 = d1.get_node()
        valid_action, valid_topo, valid_geo = False, False, False

        if action[0] == Actions.FLIP.value:
            valid_action, valid_topo, valid_geo = flip_edge(self.mesh, n1, n2)
        elif action[0] == Actions.SPLIT.value:
            valid_action, valid_topo, valid_geo = split_edge(self.mesh, n1, n2)
        elif action[0] == Actions.COLLAPSE.value:
            valid_action, valid_topo, valid_geo = collapse_edge(self.mesh, n1, n2)
        elif action[0] == Actions.CLEANUP.value:
            valid_action, valid_topo, valid_geo = cleanup_edge(self.mesh, n1, n2)
        else:
            raise ValueError("Action not defined")

        self.observation_count.register_observation(self.observation)
        if valid_action:
            # An episode is done if the actual score is the same as the ideal
            next_nodes_score, self.next_mesh_score, _, next_nodes_adjacency = global_score(self.mesh)
            terminated = np.array_equal(self._ideal_score, self.next_mesh_score)
            mesh_reward = (self._mesh_score - self.next_mesh_score)*10
            reward = mesh_reward
            self._nodes_scores, self._mesh_score, self._nodes_adjacency = next_nodes_score, self.next_mesh_score, next_nodes_adjacency
            self.observation = self._get_obs()
            self.nb_invalid_actions = 0
        elif not valid_topo:
            reward = -10
            mesh_reward = 0
            terminated = False
            self.nb_invalid_actions += 1
        elif not valid_geo:
            mesh_reward = 0
            terminated = False
            reward = 0
            self.nb_invalid_actions += 1
        else:
            raise ValueError("Invalid action")
        if self.nb_invalid_actions > 10 :
            truncated = isTruncated(self.mesh, self.darts_selected)
        else:
            truncated = False
        valid_act = valid_action, valid_topo, valid_geo
        info = self._get_info(terminated, valid_act, action, mesh_reward)

        return self.observation, reward, terminated, truncated, info
