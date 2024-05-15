import unittest
import model.mesh_struct.mesh as mesh
import numpy.testing
from model.random_trimesh import regular_mesh, random_mesh

from actions.triangular_actions import split_edge, flip_edge


class TestRandomTrimesh(unittest.TestCase):

    def test_regular_trimesh(self):
        m = regular_mesh(44)
        self.assertEqual(m.nb_nodes(), 44)
        m = regular_mesh(30)
        self.assertEqual(m.nb_nodes(), 30)
        m = regular_mesh(60)
        self.assertEqual(m.nb_nodes(), 60)

    def test_random_trimesh(self):
        m = random_mesh(44)
        self.assertEqual(m.nb_nodes(), 44)
        m = random_mesh(30)
        self.assertEqual(m.nb_nodes(), 30)
        m = random_mesh(60)
        self.assertEqual(m.nb_nodes(), 60)



if __name__ == '__main__':
    unittest.main()
