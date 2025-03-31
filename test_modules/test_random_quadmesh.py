import copy
import unittest
from mesh_model.random_quadmesh import random_mesh, mesh_shuffle
from mesh_model.mesh_struct.mesh import Mesh

#from mesh_plotter.mesh_plotter import plot_mesh


class TestRandomQuadmesh(unittest.TestCase):

    def test_random_trimesh(self):
        for _ in range(10):
            m = random_mesh()
            self.assertIsInstance(m, Mesh)


    def test_mesh_suffle(self):
        m = random_mesh()
        reg_m = copy.deepcopy(m)
        mesh_shuffle(m, 15)
        self.assertIsNot(m, reg_m)



if __name__ == '__main__':
    unittest.main()
