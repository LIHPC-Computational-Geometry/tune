import pygame
import sys
from pygame import math
import warnings

edge_color_normal = pygame.Color(30, 30, 30)  # Dark Grey
vertex_color_normal = pygame.Color(255, 0, 0)  # Red
edge_color_select = pygame.Color(0, 255, 0)  # Green
vertex_color_select = pygame.Color(128, 128, 128)  # Grey
positive_irregularity_color = pygame.Color(255, 0, 255)
negative_irregularity_color = pygame.Color(0, 0, 255)


class Vertex:
    def __init__(self, idx, x, y, value=0):
        self.idx = idx
        self.x = x
        self.y = y
        self.selected = False
        self.color = vertex_color_normal
        self.obj = None
        self.value = round(value, 0)

    def switch_selection(self):
        if self.selected:
            self.selected = False
            self.color = vertex_color_normal
        else:
            self.selected = True
            self.color = vertex_color_select

    def collide_point(self, x, y):
        return self.obj.collide_point(x, y)

    def draw(self, window, font, w_data):
        x = w_data.center.x + w_data.stretch * (self.x - w_data.scene_center.x)
        y = w_data.center.y - w_data.stretch * (self.y - w_data.scene_center.y)
        if self.value != 0 :
            if self.value > 0:
                self.obj = pygame.draw.circle(window, positive_irregularity_color,
                                          (x, y),
                                          15, 15)
                text = font.render(str(self.value), True, (255, 255, 255))
                self.blit = window.blit(text, text.get_rect(center=(x, y)))
            else:
                self.obj = pygame.draw.circle(window, negative_irregularity_color,
                                              (x, y),
                                              15, 15)
                text = font.render(str(self.value), True, (255, 255, 255))
                self.blit = window.blit(text, text.get_rect(center=(x, y)))


class Edge:

    def __init__(self, v1, v2):
        self.start = v1
        self.end = v2
        self.selected = False
        self.color = edge_color_normal
        self.obj = None

    # static method to facilitate unit test
    @staticmethod
    def is_pt_on_segment(x1, y1, x2, y2, pt, tolerance) -> bool:
        nv = pygame.math.Vector2(y1 - y2, x2 - x1)
        lp = pygame.math.Vector2(x1, y1)
        p = pygame.math.Vector2(pt)
        xy = pygame.math.Vector2(x2 - x1, y2 - y1)
        # distance from the straight line represented by the edge
        distance_ok = abs(nv.normalize().dot(p - lp)) < tolerance
        # on the segment ?
        segment_ok = (0 <= xy.normalize().dot(p - lp)) and (xy.normalize().dot(p - lp) <= xy.length())
        if distance_ok and segment_ok:
            return True
        return False

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
        return Edge.is_pt_on_segment(x1, y1, x2, y2, pt, w_data.edge_picking_pixel_tolerance)

    def collide_point(self, x, y, w_data):
        return self.distance_point(math.Vector2(x, y), w_data)

    def draw(self, window, w_data):
        self.obj = pygame.draw.line(window, self.color,
                                    [w_data.center.x + w_data.stretch * (self.start.x - w_data.scene_center.x),
                                     w_data.center.y - w_data.stretch * (self.start.y - w_data.scene_center.y)],
                                    [w_data.center.x + w_data.stretch * (self.end.x - w_data.scene_center.x),
                                     w_data.center.y - w_data.stretch * (self.end.y - w_data.scene_center.y)],
                                    w_data.edge_thickness)


class Graph:
    def __init__(self, vertices=[], edges=[], scores=[]):
        self.clear()
        self.update(vertices, edges, scores)

    def clear(self):
        self.vertices = []
        self.edges = []

    def update(self, vertices, edges, scores):
        for n in vertices:
            nodes_scores = scores[0]
            n_value = nodes_scores[n[0]]
            self.create_vertex(n[0], n[1], n[2], n_value)
        for e in edges:
            self.create_edge(e[0], e[1])

    def create_vertex(self, id: int, x: int, y: int, n_value) -> int:
        v = Vertex(id, x, y, n_value)
        self.add_vertex(v)
        return len(self.vertices) - 1

    def create_edge(self, i1: int, i2: int) -> int:
        n1, n2 = None, None
        for v in self.vertices:
            if v.idx == i1:
                n1 = v
            elif v.idx == i2:
                n2 = v
        if n1 is None or n2 is None:
            warnings.warn("try to create an edge between nodes not found")
        self.add_edge(Edge(n1, n2))
        return len(self.edges) - 1

    def add_vertex(self, n: Vertex) -> None:
        self.vertices.append(n)

    def add_edge(self, e):
        self.edges.append(e)

    def bounding_box(self):
        x_min = sys.float_info.max
        y_min = sys.float_info.max
        x_max = sys.float_info.min
        y_max = sys.float_info.min
        for v in self.vertices:
            x_min = min(v.x, x_min)
            y_min = min(v.y, y_min)
            x_max = max(v.x, x_max)
            y_max = max(v.y, y_max)
        return x_min, y_min, x_max, y_max
