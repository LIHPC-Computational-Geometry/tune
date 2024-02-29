import pygame
import sys
from pygame import math

edge_color_normal = pygame.Color(30, 30, 30)  # Dark Grey
node_color_normal = pygame.Color(255, 0, 0)  # Red
edge_color_select = pygame.Color(0, 255, 0)  # Green
node_color_select = pygame.Color(128, 128, 128)  # Grey


class Vertex:
    def __init__(self, x, y, value=2):
        self.x = x
        self.y = y
        self.selected = False
        self.color = node_color_normal
        self.obj = None
        self.value = value

    def switch_selection(self):
        if self.selected:
            self.selected = False
            self.color = node_color_normal
        else:
            self.selected = True
            self.color = node_color_select

    def collidepoint(self, x, y):
        return self.obj.collidepoint(x, y)

    def draw(self, window, font, w_data):
        x = w_data.center.x + w_data.stretch * (self.x - w_data.scene_center.x)
        y = w_data.center.y - w_data.stretch * (self.y - w_data.scene_center.y)
        self.obj = pygame.draw.circle(window, self.color,
                                      (x, y),
                                      w_data.node_size, 0)
        text = font.render(str(self.value), True, (255, 255, 255))
        window.blit(text, text.get_rect(center=(x,y)))

class Edge:

    def __init__(self, v1, v2):
        self.start = v1
        self.end = v2
        self.selected = False
        self.color = edge_color_normal
        self.obj = None

    def switch_selection(self):
        if self.selected:
            self.selected = False
            self.color = edge_color_normal
        else:
            self.selected = True
            self.color = edge_color_select

    def distance_point(self, pt, w_data):
        x1 = w_data.center.x + w_data.stretch * (self.start.x - w_data.scene_center.x)
        y1 = w_data.center.y - w_data.stretch * (self.start.y - w_data.scene_center.y)
        x2 = w_data.center.x + w_data.stretch * (self.end.x - w_data.scene_center.x)
        y2 = w_data.center.y - w_data.stretch * (self.end.y - w_data.scene_center.y)
        nv = pygame.math.Vector2(y1 - y2, x2 - x1)
        lp = pygame.math.Vector2(x1, y1)
        p = pygame.math.Vector2(pt)
        if abs(nv.normalize().dot(p - lp)) < w_data.edge_picking_pixel_tolerance:
            return True
        return False

    def collidepoint(self, x, y, w_data):
        return self.distance_point(math.Vector2(x, y), w_data)

    def draw(self, window, w_data):
        self.obj = pygame.draw.line(window, self.color,
                                    [w_data.center.x + w_data.stretch * (self.start.x - w_data.scene_center.x),
                                     w_data.center.y - w_data.stretch * (self.start.y - w_data.scene_center.y)],
                                    [w_data.center.x + w_data.stretch * (self.end.x - w_data.scene_center.x),
                                     w_data.center.y - w_data.stretch * (self.end.y - w_data.scene_center.y)],
                                    w_data.edge_thickness)


class Mesh:
    def __init__(self, nodes=[], edges=[]):
        self.nodes = []
        self.edges = []
        for n in nodes:
            self.create_node(n[0], n[1])
        for e in edges:
            self.create_edge(e[0], e[1])

    def create_node(self, x: int, y:int) -> int:
        n = Vertex(x, y)
        self.add_node(n)
        return len(self.nodes) - 1

    def create_edge(self, i1:int, i2:int) -> int:
        n1 = self.nodes[i1]
        n2 = self.nodes[i2]
        self.add_edge(Edge(n1, n2))
        return len(self.edges) - 1

    def add_node(self, n:Vertex) -> None:
        self.nodes.append(n)

    def add_edge(self, e):
        self.edges.append(e)

    def bounding_box(self):
        x_min = sys.float_info.max
        y_min = sys.float_info.max
        x_max = sys.float_info.min
        y_max = sys.float_info.min
        for n in self.nodes:
            if n.x < x_min:
                x_min = n.x
            elif n.x > x_max:
                x_max = n.x
            if n.y < y_min:
                y_min = n.y
            elif n.y > y_max:
                y_max = n.y
        return x_min, y_min, x_max, y_max

