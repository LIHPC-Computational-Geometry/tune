import pygame
from pygame import math
from pygame.locals import *
import sys

color1 = pygame.Color(30, 30, 30)  # Dark Grey
color2 = pygame.Color(255, 255, 255)  # White
color3 = pygame.Color(128, 128, 128)  # Grey
color4 = pygame.Color(255, 0, 0)  # Red
color5 = pygame.Color(0, 255, 0)  # Green

edge_color_normal = color1
node_color_normal = color4
edge_color_select = color5
node_color_select = color3


class window_data:
    def __init__(self):
        self.options = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        self.center = pygame.math.Vector2(250, 250)
        self.size = (500, 500)
        self.stretch = 10.0
        self.scene_center = pygame.math.Vector2(0, 0)
        self.scene_xmin = -250
        self.scene_xmax = 250
        self.scene_ymin = -250
        self.scene_ymax = 250
        self.node_size = 10
        self.edge_thickness = 2
        self.font_size = 25
        self.edge_picking_pixel_tolerance = 5


class Game:
    win_data = window_data()

    def __init__(self, mesh):
        pygame.init()
        self.window = pygame.display.set_mode(Game.win_data.size, Game.win_data.options)
        pygame.display.set_caption('TriGame')
        self.window.fill((255, 255, 255))
        self.font = pygame.font.SysFont(None, Game.win_data.font_size)
        self.clock = pygame.time.Clock()
        self.clock.tick(60)
        self.mesh = mesh
        Game.win_data.scene_xmin, Game.win_data.scene_ymin, Game.win_data.scene_xmax, Game.win_data.scene_ymax = mesh.bounding_box()
        Game.win_data.scene_center = math.Vector2((Game.win_data.scene_xmax + Game.win_data.scene_xmin) / 2.0,
                                                  (Game.win_data.scene_ymax + Game.win_data.scene_ymin) / 2.0)

    def draw(self):
        for e in self.mesh.edges:
            e.draw(self.window, Game.win_data)
        for n in self.mesh.nodes:
            n.draw(self.window, self.font, Game.win_data)

    def control_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == VIDEORESIZE or event.type == VIDEOEXPOSE:  # handles window minimising/maximising
                x, y = self.window.get_size()
                Game.win_data.center.x = x / 2
                Game.win_data.center.y = y / 2
                ratio = float(x) / float(Game.win_data.scene_xmax - Game.win_data.scene_xmin)
                ratio_y = float(y) / float(Game.win_data.scene_ymax - Game.win_data.scene_ymin)
                if ratio_y < ratio:
                    ratio = ratio_y

                Game.win_data.node_size = max(ratio / 100, 10)
                Game.win_data.stretch = 0.75 * ratio

                self.window.fill((255, 255, 255))
                pygame.display.flip()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # only edges can be selected, and just one at a time
                already_selected = False
                for e in self.mesh.edges:
                    if e.collidepoint(x, y, self.win_data) and not already_selected:
                        e.switch_selection()
                        already_selected = True

    def run(self):
        while True:
            self.control_events()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        print("TriGame is  starting!!")
