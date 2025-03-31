import unittest
import mesh_model.mesh_struct.mesh as mesh
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

        flip_edge(cmap, n00, n11)
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

        split_edge(cmap, n00, n11)
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

    def test_collapse(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        faces = [[0, 1, 2], [0, 2, 3]]
        cmap = mesh.Mesh(nodes, faces)
        plot_mesh(cmap)
        n00 = Node(cmap, 0)
        n11 = Node(cmap, 2)
        split_edge(cmap, n00, n11)
        plot_mesh(cmap)
        n5 = Node(cmap, 4)
        valid, _, _ = collapse_edge(cmap, n00, n5)
        d1_to_test = Dart(cmap, 7)
        d2_to_test = Dart(cmap, 0)

        self.assertEqual(valid, False)
        # Test possible collapse
        cmap = regular_mesh(16)
        d = Dart(cmap, 0)
        n0 = d.get_node()
        n1 = d.get_beta(1).get_node()
        valid, _, _ = collapse_edge(cmap, n0, n1)
        self.assertEqual(valid, True)

    def test_split_collapse_split(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        faces = [[0, 1, 2], [0, 2, 3]]
        cmap = mesh.Mesh(nodes, faces)
        n0 = Node(cmap, 0)
        n1 = Node(cmap, 1)
        n2 = Node(cmap, 2)
        n3 = Node(cmap, 3)
        split_edge(cmap, n0, n2)
        n4 = Node(cmap, 4)
        collapse_edge(cmap, n0, n4)
        split_edge(cmap, n0, n2)
        n5 = Node(cmap, 5)
        collapse_edge(cmap, n0, n5)
        split_edge(cmap, n4, n2)
        collapse_edge(cmap, n4, n5)
        collapse_edge(cmap, n2, n4)
        split_edge(cmap, n0, n2)
        split_edge(cmap, n0, n4)
        split_edge(cmap, n4, n3)
        split_edge(cmap, n4, n1)
        split_edge(cmap, n5, n1)
        n7 = Node(cmap, 7)
        n8 = Node(cmap, 8)
        collapse_edge(cmap, n7, n8)
        collapse_edge(cmap, n5, n7)



