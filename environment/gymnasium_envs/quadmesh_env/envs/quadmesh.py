
import copy
import pygame
import imageio
import sys

import numpy as np
import gymnasium as gym

from enum import Enum
from typing import Optional
from pygame.locals import *

from mesh_model.random_quadmesh import random_mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis.global_mesh_analysis import global_score
from mesh_model.mesh_analysis.quadmesh_analysis import isTruncated
from environment.gymnasium_envs.quadmesh_env.envs.mesh_conv import get_x
from environment.actions.quadrangular_actions import flip_edge_cntcw, flip_edge_cw, split_edge, collapse_edge, cleanup_edge
from environment.observation_register import ObservationRegistry
from view.window import window_data, graph
from mesh_display import MeshDisplay


class Actions(Enum):
    FLIP_CW = 0
    FLIP_CNTCW = 1
    SPLIT = 2
    COLLAPSE = 3
    CLEANUP = 4


class QuadMeshEnv(gym.Env):
    """
    QuadMesh environment is structured according to gymnasium and is used to topologically optimize quadrangular meshes topologically.
    The generated observations consist of a local topological view of the mesh. They are structured in the form of matrices :
        * The columns correspond to the surrounding area of a mesh dart.
        * Only the darts with the most irregularities in the surrounding area are retained.

    Based on these observations, the agent will choose from 4 different actions:
        * flip clockwise, flip an edge clockwise
        * flip counterclockwise, flip an edge counterclockwise
        * split, add a face
        * collapse, deleting a face

    These actions will generate rewards proportional to the improvement or deterioration of the mesh. If the chosen action is invalid, a penalty is returned.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            mesh=None,
            max_episode_steps: int = 50,
            n_darts_selected: int = 20,
            deep: int = 6,
            render_mode:  Optional[str] = None,
            with_degree_obs: bool = True,
            action_restriction: bool = False,
            obs_count: bool = False,
    ) -> None:

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        #If a mesh has been entered, it is used, otherwise a random mesh is generated.
        if mesh is not None:
            self.config = {"mesh": mesh}
            self.mesh = copy.deepcopy(mesh)
        else :
            self.config = {"mesh": None}
            self.mesh = random_mesh()

        #self.mesh_size = len(self.mesh.nodes)
        #self.nb_darts = len(self.mesh.dart_info)
        self._nodes_scores, self._mesh_score, self._ideal_score, self._nodes_adjacency = global_score(self.mesh)
        self._ideal_rewards = (self._mesh_score - self._ideal_score)*10 #arbitrary factor of 10 for rewards
        self.next_mesh_score = 0
        self.n_darts_selected = n_darts_selected
        self.restricted = action_restriction
        self.degree_observation = with_degree_obs
        self.nb_invalid_actions = 0
        self.max_steps = max_episode_steps
        self.episode_count = 0
        self.ep_len = 0
        self.darts_selected = [] # darts id observed
        self.deep = deep*2 if self.degree_observation else deep
        self.actions_info = {
            "n_flip_cntcw": 0,
            "n_flip_ccw": 0,
            "n_split": 0,
            "n_collapse": 0,
            "n_cleanup": 0,
        }

        # Definition of an observation register if required
        if obs_count:
            self.observation_count = True
            self.observation_registry = ObservationRegistry(self.n_darts_selected, self.deep, -6, 2)
        else:
            self.observation_count = False

        # Render
        if self.render_mode == "human":
            self.mesh_disp = MeshDisplay(self.mesh)
            self.graph = graph.Graph(self.mesh_disp.get_nodes_coordinates(), self.mesh_disp.get_edges(),
                                         self.mesh_disp.get_scores())
            self.win_data = window_data()
            self.window_size = 512  # The size of the PyGame window
            self.window = None
            self.clock = None

            self.recording = False
            self.frames = []

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-6,  # nodes min degree : -6
            high=2,  # nodes max degree : 2
            shape=(self.n_darts_selected, deep),
            dtype=np.int64
        )
        self.observation = None

        # We have 4 actions, flip clockwise, flip counterclockwise, split, collapse
        self.action_space = gym.spaces.MultiDiscrete([4, self.n_darts_selected])



    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if options is not None:
            self.mesh = options['mesh']
        elif self.config["mesh"] is not None:
            self.mesh = copy.deepcopy(self.config["mesh"])
        else:
            self.mesh = random_mesh()
        #self.nb_darts = len(self.mesh.dart_info)
        self._nodes_scores, self._mesh_score, self._ideal_score, self._nodes_adjacency = global_score(self.mesh)
        self._ideal_rewards = (self._mesh_score - self._ideal_score) * 10
        self.nb_invalid_actions = 0
        self.close()
        self.observation = self._get_obs()
        self.ep_len = 0
        info = self._get_info(terminated=False,valid_act=(None,None,None), action=(None,None), mesh_reward=None)
        self.actions_info = {
            "n_flip_cw": 0,
            "n_flip_cntcw": 0,
            "n_split": 0,
            "n_collapse": 0,
            "n_cleanup": 0,
        }

        if self.render_mode=="human":
            self._render_frame()
            self.recording = True
        else:
            self.recording = False
            self.frames = []

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
            "flip_cw": 1.0 if action[0]==Actions.FLIP_CW.value else 0.0,
            "flip_cntcw": 1.0 if action[0]==Actions.FLIP_CNTCW.value else 0.0,
            "split": 1.0 if action[0]==Actions.SPLIT.value else 0.0,
            "collapse": 1.0 if action[0]==Actions.COLLAPSE.value else 0.0,
            "cleanup": 1.0 if action[0]==Actions.CLEANUP.value else 0.0,
            "invalid_flip": 1.0 if (action[0]==Actions.FLIP_CW.value or action[0]==Actions.FLIP_CNTCW.value) and not valid_action else 0.0,
            "invalid_split": 1.0 if action[0]==Actions.SPLIT.value and not valid_action else 0.0,
            "invalid_collapse": 1.0 if action[0]==Actions.COLLAPSE.value and not valid_action else 0.0,
            "invalid_cleanup": 1.0 if action[0]==Actions.CLEANUP.value and not valid_action else 0.0,
            "mesh" : self.mesh,
            "darts_selected" : self.darts_selected,
            "observation_registry" : self.observation_registry if self.observation_count else None,
        }

    def _action_to_dart_id(self, action: np.ndarray) -> int:
        """
        Converts an action ID into the dart ID on which to perform the action
        :param action: action ID
        :return: the dart ID on which to perform the action
        """
        return self.darts_selected[int(action[1])]

    def step(self, action: np.ndarray):
        self.ep_len+=1
        dart_id = self._action_to_dart_id(action)
        d = Dart(self.mesh, dart_id)
        d1 = d.get_beta(1)
        n1 = d.get_node()
        n2 = d1.get_node()
        valid_action, valid_topo, valid_geo = False, False, False
        if action[0] == Actions.FLIP_CW.value:
            self.actions_info["n_flip_cw"] += 1
            valid_action, valid_topo, valid_geo = flip_edge_cw(self.mesh, n1, n2)
        elif action[0] == Actions.FLIP_CNTCW.value:
            self.actions_info["n_flip_cntcw"] += 1
            valid_action, valid_topo, valid_geo = flip_edge_cntcw(self.mesh, n1, n2)
        elif action[0] == Actions.SPLIT.value:
            self.actions_info["n_split"] += 1
            valid_action, valid_topo, valid_geo = split_edge(self.mesh, n1, n2)
        elif action[0] == Actions.COLLAPSE.value:
            self.actions_info["n_collapse"] += 1
            valid_action, valid_topo, valid_geo = collapse_edge(self.mesh, n1, n2)
        elif action[0] == Actions.CLEANUP.value:
            self.actions_info["n_cleanup"] += 1
            valid_action, valid_topo, valid_geo = cleanup_edge(self.mesh, n1, n2)
        else:
            raise ValueError("Action not defined")

        if self.observation_count:
            self.observation_registry.register_observation(self.observation)

        if valid_action:
            # An episode is done if the actual score is the same as the ideal
            next_nodes_score, self.next_mesh_score, _, next_nodes_adjacency = global_score(self.mesh)
            terminated = np.array_equal(self._ideal_score, self.next_mesh_score)
            if terminated:
                mesh_reward = (self._mesh_score - self.next_mesh_score)*10
                reward = mesh_reward
            else:
                mesh_reward = (self._mesh_score - self.next_mesh_score)*10
                reward = mesh_reward
            self._nodes_scores, self._mesh_score, self._nodes_adjacency = next_nodes_score, self.next_mesh_score, next_nodes_adjacency
            self.observation = self._get_obs()
            self.nb_invalid_actions = 0
        elif not valid_topo:
            reward = -3
            mesh_reward = 0
            terminated = False
            self.nb_invalid_actions += 1
        elif not valid_geo:
            mesh_reward = -1
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

        if self.render_mode == "human":
            self._render_frame()
        if terminated or self.ep_len>= self.max_steps:
            if self.recording and self.frames:
                imageio.mimsave(f"episode_{self.episode_count}.gif", self.frames, fps=1)
                print("Image recorded")
                self.episode_count +=1

        return self.observation, reward, terminated, truncated, info


    def _render_frame(self):
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.win_data.size, self.win_data.options)
            pygame.display.set_caption('QuadMesh')
            self.window.fill((255, 255, 255))
            self.font = pygame.font.SysFont(None, self.win_data.font_size)
            self.clock = pygame.time.Clock()
            self.clock.tick(60)
            self.win_data.scene_xmin, self.win_data.scene_ymin, self.win_data.scene_xmax, self.win_data.scene_ymax = self.graph.bounding_box()
            self.win_data.scene_center = pygame.math.Vector2((self.win_data.scene_xmax + self.win_data.scene_xmin) / 2.0,
                                                      (self.win_data.scene_ymax + self.win_data.scene_ymin) / 2.0)

        pygame.event.pump()
        self.window.fill((255, 255, 255))  # white
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == VIDEORESIZE or event.type == VIDEOEXPOSE:  # handles window minimising/maximising
                x, y = self.window.get_size()
                text_margin = 200
                self.win_data.center.x = (x - text_margin) / 2
                self.win_data.center.y = y / 2
                ratio = float(x - text_margin) / float(self.win_data.scene_xmax - self.win_data.scene_xmin)
                ratio_y = float(y) / float(self.win_data.scene_ymax - self.win_data.scene_ymin)
                if ratio_y < ratio:
                    ratio = ratio_y

                self.win_data.node_size = max(ratio / 100, 10)
                self.win_data.stretch = 0.75 * ratio

                self.window.fill((255, 255, 255))
                pygame.display.flip()

        self.graph.clear()
        self.mesh_disp = MeshDisplay(self.mesh)
        self.graph.update(self.mesh_disp.get_nodes_coordinates(), self.mesh_disp.get_edges(),
                          self.mesh_disp.get_scores())

        #Draw mesh
        for e in self.graph.edges:
            e.draw(self.window, self.win_data)
        for n in self.graph.vertices:
            n.draw(self.window, self.font, self.win_data)

        #Print action type
        if hasattr(self, 'actions_info'):
            x = self.window.get_width() - 150
            y_start = 100
            line_spacing = 25

            for i, (action_name, count) in enumerate(self.actions_info.items()):
                text = f"{action_name}: {count}"
                text_surface = self.font.render(text, True, (0, 0, 0))
                self.window.blit(text_surface, (x, y_start + i * line_spacing))

        self.clock.tick(60)
        pygame.time.delay(1200)
        pygame.display.flip()
        if self.recording:
            pixels = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = pixels.transpose([1,0,2])
            self.frames.append(frame)

    def close(self):
        if self.render_mode=="human" and self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
