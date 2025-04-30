import unittest
import os

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
import mesh_model.mesh_analysis.global_mesh_analysis as GMA
import mesh_model.mesh_analysis.quadmesh_analysis as QMA
from environment.actions.triangular_actions import split_edge_ids
from view.mesh_plotter.mesh_plots import plot_mesh
from mesh_model.reader import read_gmsh

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestMeshAnalysis(unittest.TestCase):

    def test_mesh_regular_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]]
        faces = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        cmap = Mesh(nodes,faces)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = GMA.global_score(cmap)
        self.assertEqual((0,0), (mesh_score, mesh_ideal_score) )

    def test_mesh_with_irregularities(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = GMA.global_score(cmap)
        self.assertIsNot((0, 0), (mesh_score,mesh_ideal_score) )

    def test_is_valid_action(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)

        #Boundary dart
        self.assertEqual(QMA.isValidAction(cmap, 20, 0), (False, True))

        # Flip test
        self.assertEqual(QMA.isValidAction(cmap, 3, 0), (True, True))
        self.assertEqual(QMA.isValidAction(cmap, 27, 0), (False, True))

        #Split test
        self.assertEqual(QMA.isValidAction(cmap, 0, 1), (True, True))
        self.assertEqual(QMA.isValidAction(cmap, 27, 1), (False, True))

        #Collapse test
        self.assertEqual(QMA.isValidAction(cmap, 0, 2), (True, True))
        plot_mesh(cmap)
        self.assertEqual(QMA.isValidAction(cmap, 27, 2), (False, True))

        #All action test
        self.assertEqual(QMA.isValidAction(cmap, 27, 3), (False, True))
        self.assertEqual(QMA.isValidAction(cmap, 9, 3), (True, True))

        #One action test
        self.assertEqual(QMA.isValidAction(cmap, 0, 4), (True, True))
        self.assertEqual(QMA.isValidAction(cmap, 9, 4), (True, True))
        self.assertEqual(QMA.isValidAction(cmap, 27, 4), (False, True))

        #Invalid action
        with self.assertRaises(ValueError):
            QMA.isValidAction(cmap, 0, 7)

    def test_isTruncated(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertFalse(QMA.isTruncated(cmap, darts_list))

        nodes = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        faces = [[0, 1, 3, 2]]
        cmap = Mesh(nodes, faces)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertTrue(QMA.isTruncated(cmap, darts_list))

if __name__ == '__main__':
    unittest.main()
