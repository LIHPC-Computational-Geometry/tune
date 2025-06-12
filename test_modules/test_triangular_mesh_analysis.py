import unittest

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_analysis.trimesh_analysis import TriMeshQualityAnalysis
from environment.actions.triangular_actions import split_edge_ids
from view.mesh_plotter.mesh_plots import plot_mesh

class TestMeshAnalysis(unittest.TestCase):

    def test_mesh_regular_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0,0.0],[-0.5,-1.0],[0.0,-2.0], [0.5,-1.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 5, 0], [0, 5, 6], [0, 6, 8], [6, 7, 8], [0, 8, 1]]
        cmap = Mesh(nodes,faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = m_analysis.global_score()
        self.assertEqual((0,0), (mesh_score, mesh_ideal_score) )

    def test_mesh_with_irregularities(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0, 2.0], [-2.0, 1.0], [-1.0, 0.0],
                 [-2.0, 0.0], [-2.0, -1.0], [-0.5, -1.0], [-1.0, -2.0], [0.0, -2.0], [1.0, -2.0],
                 [0.5, -1.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [1.0, 2.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [7, 0, 3], [7, 10, 0], [10, 14, 0], [0, 14, 1], [10, 12, 14],
                 [3, 4, 5], [3, 5, 6], [3, 6, 7], [7, 6, 8], [7, 8, 9], [7, 9, 10], [10, 9, 11], [10, 11, 12],
                 [14, 12, 13], [14, 13, 15], [1, 14, 15], [1, 15, 16], [1, 16, 17], [1, 17, 2], [2, 17, 18], [2, 18, 4]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = m_analysis.global_score()
        self.assertEqual((6, -2), (mesh_score,mesh_ideal_score) )

    def test_mesh_bad_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        nodes_score, mesh_score, mesh_ideal_score, adjacency = m_analysis.global_score()
        self.assertEqual((3, 1), (mesh_score, mesh_ideal_score))

    def test_split_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        split_edge_ids(m_analysis, 0, 2)
        split_edge_ids(m_analysis, 1, 4) # split impossible
        nodes_score, mesh_score, mesh_ideal_score, adjacency = m_analysis.global_score()
        self.assertEqual((3, 1), (mesh_score, mesh_ideal_score))

    def test_find_template_opposite_node_not_found(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        dart_to_test = Dart(cmap, 0)
        node = m_analysis.find_template_opposite_node(dart_to_test)
        self.assertEqual(node, None)
        dart_to_test = Dart(cmap, 2)
        node = m_analysis.find_template_opposite_node(dart_to_test)
        self.assertEqual(node, 3)

    def test_is_valid_action(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0, 2.0], [-2.0, 1.0], [-1.0, 0.0],
                 [-2.0, 0.0], [-2.0, -1.0], [-0.5, -1.0], [-1.0, -2.0], [0.0, -2.0], [1.0, -2.0],
                 [0.5, -1.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [1.0, 2.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [7, 0, 3], [7, 10, 0], [10, 14, 0], [0, 14, 1], [10, 12, 14],
                 [3, 4, 5], [3, 5, 6], [3, 6, 7], [7, 6, 8], [7, 8, 9], [7, 9, 10], [10, 9, 11], [10, 11, 12],
                 [14, 12, 13], [14, 13, 15], [1, 14, 15], [1, 15, 16], [1, 16, 17], [1, 17, 2], [2, 17, 18], [2, 18, 4]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        split_edge_ids(m_analysis, 0, 1)

        #Boundary dart
        self.assertEqual(m_analysis.isValidAction(25, 0), (False, True))

        # Flip test
        self.assertEqual(m_analysis.isValidAction(3, 0), (True, True))
        self.assertEqual(m_analysis.isValidAction(0, 0), (True, False))

        #Split test
        self.assertEqual(m_analysis.isValidAction(0, 1), (True, True))
        split_edge_ids(m_analysis, 1, 19)
        split_edge_ids(m_analysis, 1, 20)
        plot_mesh(m_analysis.mesh)
        self.assertEqual(m_analysis.isValidAction(20, 1), (True, True))
        split_edge_ids(m_analysis, 0, 19)
        split_edge_ids(m_analysis, 0, 22)
        split_edge_ids(m_analysis, 0, 23)
        plot_mesh(m_analysis.mesh)
        self.assertEqual(m_analysis.isValidAction(20, 1), (False, True)) #node n2 and n14 degree are >= 10

        #Collapse test
        self.assertEqual(m_analysis.isValidAction(20, 2), (True, True))
        self.assertEqual(m_analysis.isValidAction(2, 2), (False, False))

        #All action test
        self.assertEqual(m_analysis.isValidAction(2, 3), (False, False))
        self.assertEqual(m_analysis.isValidAction(26, 3), (False, False))
        self.assertEqual(m_analysis.isValidAction(9, 3), (True, True))

        #One action test
        self.assertEqual(m_analysis.isValidAction(0, 4), (True, True))
        self.assertEqual(m_analysis.isValidAction(9, 4), (True, True))
        self.assertEqual(m_analysis.isValidAction(46, 4), (False, True))

    def test_isFlipOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertFalse(m_analysis.isFlipOk(dart_to_test)[0])
        dart_to_test = Dart(cmap, 2)
        self.assertTrue(m_analysis.isFlipOk(dart_to_test)[0])

    def test_isSplitOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertEqual(m_analysis.isSplitOk(dart_to_test), (False, True))
        dart_to_test = Dart(cmap, 2)
        self.assertEqual(m_analysis.isSplitOk(dart_to_test), (True, True))

    def test_isCollapseOk(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        dart_to_test = Dart(cmap, 0)
        self.assertFalse(m_analysis.isCollapseOk(dart_to_test)[0])
        dart_to_test = Dart(cmap, 2)
        self.assertFalse(m_analysis.isCollapseOk(dart_to_test)[0])

        split_edge_ids(m_analysis, 0, 2)
        split_edge_ids(m_analysis, 0, 5)
        dart_to_test = Dart(cmap, 12)
        self.assertTrue(m_analysis.isCollapseOk(dart_to_test)[0])

    def test_isTruncated(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        faces = [[0, 1, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertTrue(m_analysis.isTruncated(darts_list))
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)
        darts_list = []
        for d_info in cmap.active_darts():
            darts_list.append(d_info[0])
        self.assertFalse(m_analysis.isTruncated(darts_list))

if __name__ == '__main__':
    unittest.main()


    # def test_valid_triangle(self):
    #     # test Lmax invalid
    #     vect_AB = (5.0, 0.0)
    #     vect_AC = (2.5, 5.0)
    #     vect_BC = (-2.5, 5.0)
    #     self.assertFalse(m_analysis.valid_triangle(vect_AB, vect_AC, vect_BC))
    #     # test invalid angles
    #     vect_AB = (3.0, 0.0)
    #     vect_AC = (1.5, 0.05)
    #     vect_BC = (-1.5, 0.05)
    #     self.assertFalse(m_analysis.valid_triangle(vect_AB, vect_AC, vect_BC))
    #     # test valid triangle
    #     vect_AB = (3.0, 0.0)
    #     vect_AC = (1.5, 3.0)
    #     vect_BC = (-1.5, 3.0)
    #     self.assertTrue(m_analysis.valid_triangle(vect_AB, vect_AC, vect_BC))
