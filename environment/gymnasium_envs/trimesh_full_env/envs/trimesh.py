import copy
import pygame
import imageio
import sys
import os

import numpy as np
import gymnasium as gym

from copy import deepcopy
from enum import Enum

from gymnasium import spaces
from pygame.locals import *


from mesh_model.random_trimesh import random_mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis.trimesh_analysis import TriMeshQualityAnalysis, TriMeshOldAnalysis
from environment.gymnasium_envs.trimesh_full_env.envs.mesh_conv import get_x
from environment.actions.triangular_actions import flip_edge, split_edge, collapse_edge, check_mesh
from view.mesh_plotter.mesh_plots import plot_mesh
from view.window import window_data, graph
from mesh_display import MeshDisplay

#from view.window import Game
#from mesh_display import MeshDisplay


class Actions(Enum):
    FLIP = 0
    SPLIT = 1
    COLLAPSE = 2


class TriMeshEnvFull(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
            self,
            mesh=None,
            mesh_size=9,
            max_episode_steps=20,
            n_darts_selected=7,
            deep=6,
            with_quality_obs=False,
            action_restriction=False,
            render_mode=None,
            analysis_type = "quality"
    ) -> None:

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # If a mesh has been entered, it is used, otherwise a random mesh is generated.
        if mesh is not None:
            self.config = {"mesh": mesh}
            self.mesh = copy.deepcopy(mesh)
            self.mesh_size = 0
        else:
            self.config = {"mesh": None}
            self.mesh_size = mesh_size
            self.mesh = random_mesh(mesh_size)
        self.analysis_type = analysis_type
        self.m_analysis = TriMeshQualityAnalysis(self.mesh) if self.analysis_type == "quality" else TriMeshOldAnalysis(self.mesh)
        self._nodes_scores, self._mesh_score, self._ideal_score, self._nodes_adjacency = self.m_analysis.global_score()
        self._ideal_rewards = (self._mesh_score - self._ideal_score)*10
        self.next_mesh_score = 0
        self.deep = deep
        self.n_darts_selected = n_darts_selected
        self.restricted = action_restriction
        self.quality_observation = with_quality_obs
        self.window_size = 512  # The size of the PyGame window
        self.g = None
        self.nb_invalid_actions = 0
        self.darts_selected = [] # darts id observed
        self.episode_count = 0
        self.ep_len = 0
        self.darts_selected = []
        self.max_steps = max_episode_steps

        self.actions_info = {
            "n_flip": 0,
            "n_split": 0,
            "n_collapse": 0,
        }

        self.observation_space = spaces.Box(
            low=-15,  # nodes min degree : 15
            high=15,  # nodes max degree : 15
            shape=(self.n_darts_selected, self.deep * 2 if self.quality_observation else self.deep),
            dtype=np.int64
        )

        self.observation = None

        # We have 3 action, flip, split, collapse
        self.action_space = spaces.MultiDiscrete([3, self.n_darts_selected])

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
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
            self._render_frame()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if options is not None:
            self.mesh = options['mesh']
        elif self.config["mesh"] is not None:
            self.mesh = copy.deepcopy(self.config["mesh"])
        else:
            self.mesh = random_mesh(self.mesh_size)

        self.m_analysis = TriMeshQualityAnalysis(self.mesh) if self.analysis_type=="quality" else TriMeshOldAnalysis(self.mesh)
        self._nodes_scores, self._mesh_score, self._ideal_score, self._nodes_adjacency = self.m_analysis.global_score()
        self._ideal_rewards = (self._mesh_score - self._ideal_score) * 10
        self.nb_invalid_actions = 0
        self.ep_len = 0
        self.close()
        self.observation = self._get_obs()
        info = self._get_info(terminated=False,valid_act=(None,None,None), action=(None,None), mesh_reward=None)

        self.actions_info = {
            "n_flip": 0,
            "n_split": 0,
            "n_collapse": 0,
        }

        if self.render_mode == "human":
            self.recording = True
            self._render_frame()
        else:
            self.recording = False

        return self.observation, info


    def _get_obs(self):
        irregularities, darts_list = get_x(self.m_analysis, self.n_darts_selected, self.deep, self.quality_observation, self.restricted, self._nodes_scores, self._nodes_adjacency)
        self.darts_selected = darts_list
        return irregularities

    def _get_info(self, terminated, valid_act, action, mesh_reward):
        valid_action, valid_topo, valid_geo = valid_act
        if self._mesh_score - self._ideal_score <0:
            raise ValueError("score impossible")
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
            "invalid_flip": 1.0 if action[0]==Actions.FLIP.value and not valid_action else 0.0,
            "invalid_split": 1.0 if action[0]==Actions.SPLIT.value and not valid_action else 0.0,
            "invalid_collapse": 1.0 if action[0]==Actions.COLLAPSE.value and not valid_action else 0.0,
            "mesh" : self.m_analysis.mesh,
            "mesh_analysis": self.m_analysis,
            "darts_selected": self.darts_selected,
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
        # before_mesh = deepcopy(self.mesh)
        if action[0] == Actions.FLIP.value:
            valid_action, valid_topo, valid_geo = flip_edge(self.m_analysis, n1, n2)
            self.actions_info["n_flip"] += 1
        elif action[0] == Actions.SPLIT.value:
            valid_action, valid_topo, valid_geo = split_edge(self.m_analysis, n1, n2)
            self.actions_info["n_split"] += 1
        elif action[0] == Actions.COLLAPSE.value:
            valid_action, valid_topo, valid_geo = collapse_edge(self.m_analysis, n1, n2)
            self.actions_info["n_collapse"] += 1
        else:
            raise ValueError("Action not defined")

        if valid_action:
            next_nodes_score, self.next_mesh_score, _, next_nodes_adjacency = self.m_analysis.global_score()
            # An episode is done if the actual score is the same as the ideal
            terminated = np.array_equal(self._ideal_score, self.next_mesh_score)
            mesh_reward = (self._mesh_score - self.next_mesh_score)*10
            # if mesh_reward == 10: # it should be impossible to improve only one irregularity
            #     b_mesh_analysis = TriMeshTopoAnalysis(before_mesh)
            #     plot_mesh(before_mesh)
            #     plot_mesh(self.mesh)
            #     bool1 = check_mesh(b_mesh_analysis)
            #     bool2 = check_mesh(self.m_analysis)
            reward = mesh_reward
            self._nodes_scores, self._mesh_score, self._nodes_adjacency = next_nodes_score, self.next_mesh_score, next_nodes_adjacency
            self.observation = self._get_obs()
            self.nb_invalid_actions = 0
        elif not valid_topo or not valid_geo:
            reward = -10
            mesh_reward = 0
            terminated = False
            self.nb_invalid_actions += 1
        else:
            raise ValueError("Invalid action")
        if self.nb_invalid_actions > 10 :
            truncated = self.m_analysis.isTruncated(self.darts_selected)
        else:
            truncated = False
        valid_act = valid_action, valid_topo, valid_geo
        info = self._get_info(terminated, valid_act, action, mesh_reward)

        if self.render_mode == "human":
            self._render_frame()
        #Saving episode rendering as gif
        if terminated or self.ep_len>= self.max_steps:
            if self.recording and self.frames:
                base_path = f"training/episode_recording/del/episode_star_{self.episode_count}"
                filename = base_path + ".gif"
                index = 1
                while os.path.exists(filename):
                    filename = f"{base_path}_{index}.gif"
                    index += 1

                imageio.mimsave(filename, self.frames, fps=1)
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
