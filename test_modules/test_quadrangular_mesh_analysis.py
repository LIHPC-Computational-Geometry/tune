import unittest
import os

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshOldAnalysis
from environment.actions.quadrangular_actions import split_edge_ids, flip_edge_cw_ids
from view.mesh_plotter.mesh_plots import plot_mesh
from mesh_model.reader import read_gmsh

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestMeshOldAnalysis(unittest.TestCase):

    def test_mesh_regular_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 3.0]]
        faces = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        cmap = Mesh(nodes,faces)
        qma = QuadMeshOldAnalysis(cmap)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = qma.global_score()
        self.assertEqual((0,0), (mesh_score, mesh_ideal_score) )

    def test_mesh_with_irregularities(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        qma = QuadMeshOldAnalysis(cmap)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = qma.global_score()
        self.assertIsNot((0, 0), (mesh_score,mesh_ideal_score) )

    def test_is_valid_action(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        qma = QuadMeshOldAnalysis(cmap)

        #Boundary dart
        self.assertEqual(qma.isValidAction(20, 0), (False, True))

        # Flip Clockwise test
        self.assertEqual(qma.isValidAction(3, 0), (True, True))
        self.assertEqual(qma.isValidAction(27, 0), (False, True))

        # Flip Counterclockwise test
        self.assertEqual(qma.isValidAction(3, 1), (True, True))
        self.assertEqual(qma.isValidAction(27, 1), (False, True))

        #Split test
        self.assertEqual(qma.isValidAction(0, 2), (True, True))
        self.assertEqual(qma.isValidAction(27, 2), (False, True))

        #Collapse test
        self.assertEqual(qma.isValidAction(0, 3), (True, True))
        plot_mesh(cmap)
        self.assertEqual(qma.isValidAction(27, 3), (False, True))

        #Cleanup test action id = 4

        #All action test
        self.assertEqual(qma.isValidAction(27, 5), (False, True))
        flip_edge_cw_ids(qma,13,37)
        self.assertEqual(qma.isValidAction(66, 5), (False, False))
        self.assertEqual(qma.isValidAction(9, 5), (True, True))

        #One action test
        self.assertEqual(qma.isValidAction(0, 6), (True, True))
        self.assertEqual(qma.isValidAction(9, 6), (True, True))
        self.assertEqual(qma.isValidAction(27, 6), (False, True))

        #Invalid action
        with self.assertRaises(ValueError):
            qma.isValidAction(0, 7)

    def test_isTruncated(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        qma = QuadMeshOldAnalysis(cmap)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertFalse(qma.isTruncated(darts_list))

        nodes = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        faces = [[0, 1, 3, 2]]
        cmap = Mesh(nodes, faces)
        qma = QuadMeshOldAnalysis(cmap)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertTrue(qma.isTruncated(darts_list))

if __name__ == '__main__':
    unittest.main()
