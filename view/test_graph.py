import unittest
from pygame import math
from graph import Edge

class TestMain(unittest.TestCase):

    def test_Edge_collide_point(self):
        tolerance = 0.0001
        p1 = math.Vector2(0.5, 0.0)
        self.assertTrue(Edge.is_pt_on_segment(0.0, 0.0, 1.0, 0.0, p1, tolerance))
        p2 = math.Vector2(1.5, 0.0)
        self.assertFalse(Edge.is_pt_on_segment(0.0, 0.0, 1.0, 0.0, p2, tolerance))

if __name__ == '__main__':
    unittest.main()
