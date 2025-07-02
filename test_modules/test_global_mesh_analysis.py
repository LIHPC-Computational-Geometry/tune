import math
import unittest
import os

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.trimesh_analysis import TriMeshQualityAnalysis, TriMeshOldAnalysis
from mesh_model.reader import read_gmsh
from view.mesh_plotter.mesh_plots import plot_mesh

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestGlobalMeshAnalysis(unittest.TestCase):

    def test_angle_by_coord(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        m_analysis = TriMeshOldAnalysis(cmap)
        self.assertEqual(m_analysis.get_angle_by_coord(1,0,0,0,0,1), 90)
        self.assertEqual(m_analysis.get_angle_by_coord(-1,0,0,0,1,0), 180)

    def test_angle_from_sides(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        m_analysis = TriMeshOldAnalysis(cmap)
        self.assertAlmostEquals(math.degrees(m_analysis.angle_from_sides(1,1,1)), 60)
        self.assertAlmostEquals(math.degrees(m_analysis.angle_from_sides(0.00000001, 1, 1)), 0)
        self.assertAlmostEquals(math.degrees(m_analysis.angle_from_sides(1, 0.5, 0.5)), 180)

        with self.assertRaises(ValueError):
            m_analysis.angle_from_sides(1.1, 0.5, 0.5)

if __name__ == '__main__':
    unittest.main()