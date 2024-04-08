import pygame
import sys
from pygame import math

edge_color_normal = pygame.Color(30, 30, 30)  # Dark Grey
vertex_color_normal = pygame.Color(255, 0, 0)  # Red
edge_color_select = pygame.Color(0, 255, 0)  # Green
vertex_color_select = pygame.Color(128, 128, 128)  # Grey


class Vertex:
    def __init__(self, idx, x, y, value=0):
        self.idx = idx
        self.x = x
        self.y = y
        self.selected = False
        self.color = vertex_color_normal
        self.obj = None
        self.value = value

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
        self.obj = pygame.draw.circle(window, self.color,
                                      (x, y),
                                      w_data.node_size, 0)
        text = font.render(str(self.value), True, (255, 255, 255))
        window.blit(text, text.get_rect(center=(x, y)))


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
        segment_ok = xy.normalize().dot(p - lp) < xy.length()
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
    def __init__(self, vertices=[], edges=[]):
        self.clear()
        self.update(vertices, edges)

    def clear(self):
        self.vertices = []
        self.edges = []

    def update(self, vertices, edges):
        for idx, n in enumerate(vertices):
            self.create_vertex(idx, n[0], n[1])
        for e in edges:
            self.create_edge(e[0], e[1])

    def create_vertex(self, id: int, x: int, y: int) -> int:
        v = Vertex(id, x, y)
        self.add_vertex(v)
        return len(self.vertices) - 1

    def create_edge(self, i1: int, i2: int) -> int:
        n1 = self.vertices[i1]
        n2 = self.vertices[i2]
        n1.value += 1
        n2.value += 1
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
            if v.x < x_min:
                x_min = v.x
            elif v.x > x_max:
                x_max = v.x
            if v.y < y_min:
                y_min = v.y
            elif v.y > y_max:
                y_max = v.y
        return x_min, y_min, x_max, y_max
