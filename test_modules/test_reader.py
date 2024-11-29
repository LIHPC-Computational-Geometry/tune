import unittest
from mesh_model.reader import read_medit
from mesh_model.reader import read_gmsh

import os

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestReader(unittest.TestCase):

    def test_read_medit(self):
        filename = os.path.join(TESTFILE_FOLDER, 'circle_coarse.mesh')
        m = read_medit(filename)
        self.assertEqual(m.nb_nodes(), 98)
        self.assertEqual(m.nb_faces(), 164)

    def test_read_gmsh_tri(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_tri.msh')
        m = read_gmsh(filename)
        self.assertEqual(m.nb_nodes(), 40)
        self.assertEqual(m.nb_faces(), 56)

    def test_read_gmsh_quad(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        m = read_gmsh(filename)
        self.assertEqual(m.nb_nodes(), 40)
        self.assertEqual(m.nb_faces(), 28)


if __name__ == '__main__':
    unittest.main()
