import unittest

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
import mesh_model.mesh_analysis as Mesh_analysis
from actions.triangular_actions import split_edge_ids
from plots.mesh_plotter import plot_mesh

class TestMeshAnalysis(unittest.TestCase):

    def test_mesh_regular_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0,0.0],[-0.5,-1.0],[0.0,-2.0], [0.5,-1.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 5, 0], [0, 5, 6], [0, 6, 8], [6, 7, 8], [0, 8, 1]]
        cmap = Mesh(nodes,faces)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((0,0), (mesh_score,mesh_ideal_score) )  # add assertion here

    def test_mesh_with_irregularities(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0, 2.0], [-2.0, 1.0], [-1.0, 0.0],
                 [-2.0, 0.0], [-2.0, -1.0], [-0.5, -1.0], [-1.0, -2.0], [0.0, -2.0], [1.0, -2.0],
                 [0.5, -1.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [1.0, 2.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [7, 0, 3], [7, 10, 0], [10, 14, 0], [0, 14, 1], [10, 12, 14],
                 [3, 4, 5], [3, 5, 6], [3, 6, 7], [7, 6, 8], [7, 8, 9], [7, 9, 10], [10, 9, 11], [10, 11, 12],
                 [14, 12, 13], [14, 13, 15], [1, 14, 15], [1, 15, 16], [1, 16, 17], [1, 17, 2], [2, 17, 18], [2, 18, 4]]
        cmap = Mesh(nodes, faces)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((6, -2), (mesh_score,mesh_ideal_score) )  # add assertion here

    def test_mesh_bad_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((3, 1), (mesh_score, mesh_ideal_score))

    def test_split_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        split_edge_ids(cmap, 0, 2)
        split_edge_ids(cmap, 1, 2) # split impossible
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((3, 1), (mesh_score, mesh_ideal_score))

    def test_find_template_opposite_node_not_found(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        dart_to_test = Dart(cmap, 0)
        node = Mesh_analysis.find_template_opposite_node(dart_to_test)
        self.assertEqual(node, None)
        dart_to_test = Dart(cmap, 2)
        node = Mesh_analysis.find_template_opposite_node(dart_to_test)
        self.assertEqual(node, 3)

    def test_is_valid_action(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        self.assertEqual(Mesh_analysis.isValidAction(cmap, 0, 3), False)
        self.assertEqual(Mesh_analysis.isValidAction(cmap, 2, 0), True)
        self.assertEqual(Mesh_analysis.isValidAction(cmap, 2, 3), False)

    def test_isFlipOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        dart_to_test = Dart(cmap, 0)
        self.assertFalse(Mesh_analysis.isFlipOk(dart_to_test))
        dart_to_test = Dart(cmap, 2)
        self.assertTrue(Mesh_analysis.isFlipOk(dart_to_test))

    def test_isSplitOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        plot_mesh(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertFalse(Mesh_analysis.isSplitOk(dart_to_test))
        dart_to_test = Dart(cmap, 2)
        self.assertTrue(Mesh_analysis.isSplitOk(dart_to_test))

    def test_isCollapseOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        plot_mesh(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertFalse(Mesh_analysis.isCollapseOk(dart_to_test))
        dart_to_test = Dart(cmap, 2)
        self.assertFalse(Mesh_analysis.isCollapseOk(dart_to_test))


if __name__ == '__main__':
    unittest.main()
