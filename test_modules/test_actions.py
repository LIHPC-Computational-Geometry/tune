import unittest
import scipy
import mesh_model.mesh_struct.mesh as mesh
from mesh_model.mesh_analysis.trimesh_analysis import TriMeshQualityAnalysis
from mesh_model.mesh_struct.mesh_elements import Dart, Node
from mesh_model.random_trimesh import regular_mesh
from environment.actions.triangular_actions import split_edge, flip_edge, collapse_edge
from view.mesh_plotter.mesh_plots import plot_mesh


class TestActions(unittest.TestCase):

    def test_flip(self):
        cmap = mesh.Mesh()
        n00 = cmap.add_node(0, 0)
        n01 = cmap.add_node(0, 1)
        n10 = cmap.add_node(1, 0)
        n11 = cmap.add_node(1, 1)

        t1 = cmap.add_triangle(n00, n10, n11)
        t2 = cmap.add_triangle(n00, n11, n01)

        d1 = t1.get_dart()
        # d1 goes from n00 to n10
        self.assertEqual(d1.get_node(), n00)
        d1 = d1.get_beta(1).get_beta(1)
        # now d1 goes from n11 to n00
        self.assertEqual(d1.get_node(), n11)

        d2 = t2.get_dart()  # goes from n00 to n11
        self.assertEqual(d2.get_node(), n00)
        # We sew on both directions
        d1.set_beta(2, d2)
        d2.set_beta(2, d1)

        m_analysis = TriMeshQualityAnalysis(cmap)
        done = flip_edge(m_analysis, n00, n11)

        self.assertTrue(done[0])
        self.assertEqual(2, cmap.nb_faces())
        self.assertEqual(4, cmap.nb_nodes())

    def test_split(self):
        cmap = mesh.Mesh()
        n00 = cmap.add_node(0, 0)
        n01 = cmap.add_node(0, 1)
        n10 = cmap.add_node(1, 0)
        n11 = cmap.add_node(1, 1)

        t1 = cmap.add_triangle(n00, n10, n11)
        t2 = cmap.add_triangle(n00, n11, n01)

        cmap.set_twin_pointers()
        m_analysis = TriMeshQualityAnalysis(cmap)

        done = split_edge(m_analysis, n00, n11)
        n_new = Node(cmap, 4)

        self.assertTrue(done[0])
        d1 = t1.get_dart()
        # d1 goes from nnew to n10
        self.assertEqual(d1.get_node(), n_new)
        d1 = d1.get_beta(1).get_beta(1)
        # now d1 goes from n11 to n00
        self.assertEqual(d1.get_node(), n11)

        d2 = t2.get_dart()  # goes from n00 to n11
        self.assertEqual(d2.get_node(), n00)
        self.assertEqual(4, cmap.nb_faces())
        self.assertEqual(5, cmap.nb_nodes())

    def test_collapse(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        faces = [[0, 1, 2], [0, 2, 3]]
        cmap = mesh.Mesh(nodes, faces)
        n00 = Node(cmap, 0)
        n11 = Node(cmap, 2)
        m_analysis = TriMeshQualityAnalysis(cmap)

        split_edge(m_analysis, n00, n11)
        n5 = Node(cmap, 4)
        split_edge(m_analysis, n11, n5)
        n6 = Node(cmap, 5)
        plot_mesh(cmap)

        #Collapse not possible
        valid, _, _ = collapse_edge(m_analysis, n11, n6)
        self.assertEqual(valid, False)

        # Collapse possible
        valid, _, _ = collapse_edge(m_analysis, n6, n5)
        self.assertEqual(valid, True)
        plot_mesh(cmap)


    def test_split_collapse_split(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        faces = [[0, 1, 2], [0, 2, 3]]
        cmap = mesh.Mesh(nodes, faces)
        m_analysis = TriMeshQualityAnalysis(cmap)

        n0 = Node(cmap, 0)
        n1 = Node(cmap, 1)
        n2 = Node(cmap, 2)
        n3 = Node(cmap, 3)
        split_edge(m_analysis, n0, n2)
        n4 = Node(cmap, 4)
        collapse_edge(m_analysis, n0, n4)
        split_edge(m_analysis, n0, n2)
        n5 = Node(cmap, 5)
        collapse_edge(m_analysis, n0, n5)
        split_edge(m_analysis, n4, n2)
        collapse_edge(m_analysis, n4, n5)
        collapse_edge(m_analysis, n2, n4)
        split_edge(m_analysis, n0, n2)
        split_edge(m_analysis, n0, n4)
        split_edge(m_analysis, n4, n3)
        split_edge(m_analysis, n4, n1)
        split_edge(m_analysis, n5, n1)
        n7 = Node(cmap, 7)
        n8 = Node(cmap, 8)
        collapse_edge(m_analysis, n7, n8)
        collapse_edge(m_analysis, n5, n7)


if __name__ == '__main__':
    unittest.main()