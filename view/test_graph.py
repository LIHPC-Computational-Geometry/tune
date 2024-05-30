import unittest
from pygame import math
from graph import Edge
from graph import Vertex
from graph import Graph


class TestGraph(unittest.TestCase):

    def test_Edge_collide_point(self):
        tolerance = 0.0001
        p1 = math.Vector2(0.5, 0.0)
        self.assertTrue(Edge.is_pt_on_segment(0.0, 0.0, 1.0, 0.0, p1, tolerance))
        self.assertFalse(Edge.is_pt_on_segment(1.0, 0.0, 2.0, 0.0, p1, tolerance))
        p2 = math.Vector2(1.5, 0.0)
        self.assertFalse(Edge.is_pt_on_segment(0.0, 0.0, 1.0, 0.0, p2, tolerance))

    def test_bounding_box(self):
        g = Graph()
        g.add_vertex(Vertex(0, 2.0, 3.0))
        self.assertEqual((2.0, 3.0, 2.0, 3.0), g.bounding_box())
        g.add_vertex(Vertex(1, 1.0, 2.5))
        g.add_vertex(Vertex(2, 0.0, 2.1))
        self.assertEqual((0.0, 2.1, 2.0, 3.0), g.bounding_box())

if __name__ == '__main__':
    unittest.main()
