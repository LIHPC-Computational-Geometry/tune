import copy
import unittest
from mesh_model.random_trimesh import regular_mesh, random_mesh, random_flip_mesh, mesh_shuffle
from mesh_model.mesh_struct.mesh import Mesh

#from mesh_plotter.mesh_plotter import plot_mesh


class TestRandomTrimesh(unittest.TestCase):

    def test_regular_trimesh(self):
        m = regular_mesh(44)
        self.assertEqual(m.nb_nodes(), 44)
        m = regular_mesh(30)
        self.assertEqual(m.nb_nodes(), 30)
        m = regular_mesh(60)
        self.assertEqual(m.nb_nodes(), 60)

    def test_random_trimesh(self):
        for _ in range(10):
            m = random_mesh(30)
            self.assertIsInstance(m, Mesh)

    def test_random_flip_mesh(self):
        m = random_flip_mesh(44)
        self.assertEqual(m.nb_nodes(), 44)
        m = random_flip_mesh(30)
        self.assertEqual(m.nb_nodes(), 30)
        m = random_flip_mesh(60)
        self.assertEqual(m.nb_nodes(), 60)

    def test_mesh_suffle(self):
        m = regular_mesh(15)
        reg_m = copy.deepcopy(m)
        mesh_shuffle(m, 15)
        self.assertIsNot(m, reg_m)



if __name__ == '__main__':
    unittest.main()
