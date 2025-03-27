import unittest

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
import mesh_model.mesh_analysis.global_mesh_analysis as GMA
import mesh_model.mesh_analysis.trimesh_analysis as TMA
import mesh_model.mesh_analysis.quadmesh_analysis as QMA
from environment.actions.triangular_actions import split_edge_ids
from view.mesh_plotter.mesh_plots import plot_mesh

class TestMeshAnalysis(unittest.TestCase):

    def test_mesh_regular_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0,0.0],[-0.5,-1.0],[0.0,-2.0], [0.5,-1.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 5, 0], [0, 5, 6], [0, 6, 8], [6, 7, 8], [0, 8, 1]]
        cmap = Mesh(nodes,faces)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = GMA.global_score(cmap)
        self.assertEqual((0,0), (mesh_score, mesh_ideal_score) )

    def test_mesh_with_irregularities(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0, 2.0], [-2.0, 1.0], [-1.0, 0.0],
                 [-2.0, 0.0], [-2.0, -1.0], [-0.5, -1.0], [-1.0, -2.0], [0.0, -2.0], [1.0, -2.0],
                 [0.5, -1.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [1.0, 2.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [7, 0, 3], [7, 10, 0], [10, 14, 0], [0, 14, 1], [10, 12, 14],
                 [3, 4, 5], [3, 5, 6], [3, 6, 7], [7, 6, 8], [7, 8, 9], [7, 9, 10], [10, 9, 11], [10, 11, 12],
                 [14, 12, 13], [14, 13, 15], [1, 14, 15], [1, 15, 16], [1, 16, 17], [1, 17, 2], [2, 17, 18], [2, 18, 4]]
        cmap = Mesh(nodes, faces)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = GMA.global_score(cmap)
        self.assertEqual((6, -2), (mesh_score,mesh_ideal_score) )

    def test_mesh_bad_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = GMA.global_score(cmap)
        self.assertEqual((3, 1), (mesh_score, mesh_ideal_score))

    def test_split_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        split_edge_ids(cmap, 0, 2)
        split_edge_ids(cmap, 1, 2) # split impossible
        nodes_score, mesh_score, mesh_ideal_score, adjacency = GMA.global_score(cmap)
        self.assertEqual((3, 1), (mesh_score, mesh_ideal_score))

    def test_find_template_opposite_node_not_found(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        dart_to_test = Dart(cmap, 0)
        node = TMA.find_template_opposite_node(dart_to_test)
        self.assertEqual(node, None)
        dart_to_test = Dart(cmap, 2)
        node = TMA.find_template_opposite_node(dart_to_test)
        self.assertEqual(node, 3)

    def test_is_valid_action(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0, 2.0], [-2.0, 1.0], [-1.0, 0.0],
                 [-2.0, 0.0], [-2.0, -1.0], [-0.5, -1.0], [-1.0, -2.0], [0.0, -2.0], [1.0, -2.0],
                 [0.5, -1.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [1.0, 2.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [7, 0, 3], [7, 10, 0], [10, 14, 0], [0, 14, 1], [10, 12, 14],
                 [3, 4, 5], [3, 5, 6], [3, 6, 7], [7, 6, 8], [7, 8, 9], [7, 9, 10], [10, 9, 11], [10, 11, 12],
                 [14, 12, 13], [14, 13, 15], [1, 14, 15], [1, 15, 16], [1, 16, 17], [1, 17, 2], [2, 17, 18], [2, 18, 4]]
        cmap = Mesh(nodes, faces)
        split_edge_ids(cmap, 0, 1)

        #Boundary dart
        self.assertEqual(TMA.isValidAction(cmap, 25, 0), (False, True))

        # Flip test
        self.assertEqual(TMA.isValidAction(cmap, 3, 0), (True, True))
        self.assertEqual(TMA.isValidAction(cmap, 0, 0), (False, True))

        #Split test
        self.assertEqual(TMA.isValidAction(cmap, 0, 1), (True, True))
        split_edge_ids(cmap, 1, 19)
        split_edge_ids(cmap, 1, 20)
        self.assertEqual(TMA.isValidAction(cmap, 20, 1), (True, False))
        split_edge_ids(cmap, 0, 19)
        split_edge_ids(cmap, 0, 22)
        split_edge_ids(cmap, 0, 23)
        self.assertEqual(TMA.isValidAction(cmap, 20, 1), (False, True))

        #Collapse test
        self.assertEqual(TMA.isValidAction(cmap, 20, 2), (True, True))
        plot_mesh(cmap)
        self.assertEqual(TMA.isValidAction(cmap, 2, 2), (False, True))

        #All action test
        self.assertEqual(TMA.isValidAction(cmap, 2, 3), (False, False))
        self.assertEqual(TMA.isValidAction(cmap, 26, 3), (False, False))
        self.assertEqual(TMA.isValidAction(cmap, 9, 3), (True, True))

        #One action test
        self.assertEqual(TMA.isValidAction(cmap, 0, 4), (True, True))
        self.assertEqual(TMA.isValidAction(cmap, 9, 4), (True, True))
        self.assertEqual(TMA.isValidAction(cmap, 94, 4), (False, False))

    def test_isFlipOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        plot_mesh(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertFalse(TMA.isFlipOk(dart_to_test)[0])
        dart_to_test = Dart(cmap, 2)
        self.assertTrue(TMA.isFlipOk(dart_to_test))

    def test_isSplitOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        plot_mesh(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertEqual(TMA.isSplitOk(dart_to_test), (False, True))
        dart_to_test = Dart(cmap, 2)
        self.assertEqual(TMA.isSplitOk(dart_to_test), (True, True))

    def test_isCollapseOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        plot_mesh(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertFalse(TMA.isCollapseOk(dart_to_test)[0])
        dart_to_test = Dart(cmap, 2)
        self.assertFalse(TMA.isCollapseOk(dart_to_test)[0])

    def test_valid_triangle(self):
        # test Lmax invalid
        vect_AB = (5.0, 0.0)
        vect_AC = (2.5, 5.0)
        vect_BC = (-2.5, 5.0)
        self.assertFalse(TMA.valid_triangle(vect_AB, vect_AC, vect_BC))
        # test invalid angles
        vect_AB = (3.0, 0.0)
        vect_AC = (1.5, 0.05)
        vect_BC = (-1.5, 0.05)
        self.assertFalse(TMA.valid_triangle(vect_AB, vect_AC, vect_BC))
        # test valid triangle
        vect_AB = (3.0, 0.0)
        vect_AC = (1.5, 3.0)
        vect_BC = (-1.5, 3.0)
        self.assertTrue(TMA.valid_triangle(vect_AB, vect_AC, vect_BC))

    def test_isTruncated(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        faces = [[0, 1, 2]]
        cmap = Mesh(nodes, faces)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertTrue(TMA.isTruncated(cmap, darts_list))
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertFalse(TMA.isTruncated(cmap, darts_list))

if __name__ == '__main__':
    unittest.main()
