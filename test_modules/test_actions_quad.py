import unittest
import os
import mesh_model.mesh_struct.mesh as mesh
from mesh_model.mesh_struct.mesh_elements import Dart, Node
from mesh_model.random_quadmesh import random_mesh
from environment.actions.quadrangular_actions import flip_edge_cntcw, flip_edge_cw, split_edge, collapse_edge, cleanup_edge
from view.mesh_plotter.mesh_plots import plot_mesh
from mesh_model.reader import read_gmsh

TESTFILE_FOLDER = os.path.join(os.path.dirname(__file__), '../mesh_files/')

class TestQuadActions(unittest.TestCase):

    def test_flip(self):
        cmap = mesh.Mesh()
        n00 = cmap.add_node(0, 0)
        n01 = cmap.add_node(0, 1)
        n10 = cmap.add_node(1, 0)
        n11 = cmap.add_node(1, 1)
        n20 = cmap.add_node(2, 0)
        n21 = cmap.add_node(2, 1)

        q1 = cmap.add_quad(n11, n10, n20, n21)
        q2 = cmap.add_quad(n10, n11, n01, n00)
        cmap.set_twin_pointers()
        plot_mesh(cmap)

        d0 = q1.get_dart()
        # d1 goes from n11 to n10
        self.assertEqual(d0.get_node(), n11)

        d2 = q2.get_dart()  # goes from n10 to n11
        self.assertEqual(d2.get_node(), n10)


        self.assertEqual(flip_edge_cntcw(cmap, n11, n10), (True,True,True))
        self.assertEqual(2, cmap.nb_faces())
        self.assertEqual(6, cmap.nb_nodes())
        plot_mesh(cmap)
        self.assertFalse(flip_edge_cntcw(cmap, n11, n10)[0])

    def test_split(self):
        cmap = mesh.Mesh()
        n00 = cmap.add_node(0, 0)
        n01 = cmap.add_node(0, 1)
        n02 = cmap.add_node(0, 2)
        n10 = cmap.add_node(1, 0)
        n11 = cmap.add_node(1, 1)
        n12 = cmap.add_node(1, 2)
        n20 = cmap.add_node(2, 0)
        n21 = cmap.add_node(2, 1)
        n22 = cmap.add_node(2, 2)

        q1 = cmap.add_quad(n00, n10, n11, n01)
        q2 = cmap.add_quad(n10, n20, n21, n11)
        q3 = cmap.add_quad(n11, n21, n22, n12)
        q4 = cmap.add_quad(n01, n11, n12, n02)
        cmap.set_twin_pointers()
        plot_mesh(cmap)
        found, d = cmap.find_inner_edge(n11, n21)
        self.assertTrue(found)
        self.assertEqual(split_edge(cmap, n11, n21), (True,True,True))
        self.assertEqual(10, cmap.nb_nodes())
        self.assertEqual(5, cmap.nb_faces())
        plot_mesh(cmap)
        self.assertFalse(split_edge(cmap, n20, n21)[0])

    def test_collapse(self):
        cmap = mesh.Mesh()
        n00 = cmap.add_node(0, 0)
        n01 = cmap.add_node(0, 1)
        n02 = cmap.add_node(0, 2)
        n10 = cmap.add_node(1, 0)
        n051 = cmap.add_node(0.75, 1)
        n151 = cmap.add_node(1.25, 1)
        n12 = cmap.add_node(1, 2)
        n20 = cmap.add_node(2, 0)
        n21 = cmap.add_node(2, 1)
        n22 = cmap.add_node(2, 2)

        q1 = cmap.add_quad(n00, n10, n051, n01)
        q2 = cmap.add_quad(n10, n20, n21, n151)
        q3 = cmap.add_quad(n151, n21, n22, n12)
        q4 = cmap.add_quad(n01, n051, n12, n02)
        q5 = cmap.add_quad(n051, n10, n151, n12)
        cmap.set_twin_pointers()
        plot_mesh(cmap)
        found, d = cmap.find_inner_edge(n151, n12)
        self.assertTrue(found)
        plot_mesh(cmap)
        self.assertEqual(collapse_edge(cmap, n151, n12), (True, True, True))
        self.assertEqual(9, cmap.nb_nodes())
        self.assertEqual(4, cmap.nb_faces())
        plot_mesh(cmap)
        self.assertFalse(split_edge(cmap, n20, n21)[0])

    def test_cleanup(self):
        cmap = mesh.Mesh()
        n00 = cmap.add_node(0, 0)
        n01 = cmap.add_node(0, 1)
        n02 = cmap.add_node(0, 2)
        n10 = cmap.add_node(1, 0)
        n051 = cmap.add_node(0.75, 1)
        n151 = cmap.add_node(1.25, 1)
        n12 = cmap.add_node(1, 2)
        n20 = cmap.add_node(2, 0)
        n21 = cmap.add_node(2, 1)
        n22 = cmap.add_node(2, 2)

        q1 = cmap.add_quad(n00, n10, n051, n01)
        q2 = cmap.add_quad(n10, n20, n21, n151)
        q3 = cmap.add_quad(n151, n21, n22, n12)
        q4 = cmap.add_quad(n01, n051, n12, n02)
        q5 = cmap.add_quad(n051, n10, n151, n12)
        cmap.set_twin_pointers()
        plot_mesh(cmap)
        found, d = cmap.find_inner_edge(n151, n12)
        self.assertTrue(found)
        self.assertEqual(cleanup_edge(cmap, n151, n21), (True, True, True))
        self.assertEqual(7, cmap.nb_nodes())
        self.assertEqual(3, cmap.nb_faces())
        plot_mesh(cmap)


    def test_actions(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        plot_mesh(cmap)
        d = Dart(cmap, 14)
        n1= d.get_node()
        n2 = (d.get_beta(1)).get_node()
        self.assertEqual(collapse_edge(cmap, n1, n2), (True,True,True))
        plot_mesh(cmap)
        d = Dart(cmap, 32)
        n1 = d.get_node()
        n2 = (d.get_beta(1)).get_node()
        self.assertEqual(flip_edge_cntcw(cmap, n1, n2), (True,True,True))

        plot_mesh(cmap)

    def test_random_quad(self):
        filename = os.path.join(TESTFILE_FOLDER, 't1_quad.msh')
        cmap = read_gmsh(filename)
        plot_mesh(cmap)
        mesh = random_mesh()
        plot_mesh(mesh)


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

    def test_simple_mesh(self):
        filename = os.path.join(TESTFILE_FOLDER, 'simple_quad.msh')
        cmap = read_gmsh(filename)
        self.assertEqual(6, cmap.nb_faces())
        self.assertEqual(11, cmap.nb_nodes())

        #Collapse node 10 (0.67,0.67) from edge 10-6
        collapse_edge(cmap, Node(cmap, 10), Node(cmap, 6))
        self.assertEqual(5, cmap.nb_faces())
        self.assertEqual(10, cmap.nb_nodes())
        self.assertTrue(Node(cmap,10).get_dart().id < 0)
        plot_mesh(cmap)
        #Flip edge 5-3
        self.assertTrue(cmap.find_inner_edge(Node(cmap, 5), Node(cmap, 3))[0])
        flip_edge_cntcw(cmap, Node(cmap, 5), Node(cmap, 3))
        self.assertFalse(cmap.find_inner_edge(Node(cmap, 5), Node(cmap, 3))[0])
        plot_mesh(cmap)
        #Collapse node 9 (0.33, 0.33) from edge 9-4
        collapse_edge(cmap, Node(cmap, 9), Node(cmap, 4))
        self.assertEqual(4, cmap.nb_faces())
        self.assertEqual(9, cmap.nb_nodes())
        self.assertTrue(Node(cmap, 9).get_dart().id < 0)
        plot_mesh(cmap)
        #Flip edge 7-1
        self.assertTrue(cmap.find_inner_edge(Node(cmap, 7), Node(cmap, 1))[0])
        flip_edge_cntcw(cmap, Node(cmap, 7), Node(cmap, 1))
        self.assertFalse(cmap.find_inner_edge(Node(cmap, 7), Node(cmap, 1))[0])
        plot_mesh(cmap)
        #Flip edge 3-8
        self.assertTrue(cmap.find_inner_edge(Node(cmap, 3), Node(cmap, 8))[0])
        flip_edge_cntcw(cmap, Node(cmap, 3), Node(cmap, 8))
        self.assertFalse(cmap.find_inner_edge(Node(cmap, 3), Node(cmap, 8))[0])
        plot_mesh(cmap)
        #Flip edge 1-8
        self.assertTrue(cmap.find_inner_edge(Node(cmap, 1), Node(cmap, 8))[0])
        flip_edge_cntcw(cmap, Node(cmap, 1), Node(cmap, 8))
        self.assertFalse(cmap.find_inner_edge(Node(cmap, 1), Node(cmap, 8))[0])
        plot_mesh(cmap)
        #Split edge 2-7 and create new node n9 at coordinate (0.5, 0.75)
        split_edge(cmap, Node(cmap, 2), Node(cmap, 7))
        self.assertEqual(5, cmap.nb_faces())
        self.assertEqual(10, cmap.nb_nodes())
        self.assertTrue(Node(cmap, 9).get_dart().id > 0)
        plot_mesh(cmap)
        # Split edge 0-5 and create new node n10 at coordinate (0.5, 0.25)
        split_edge(cmap, Node(cmap, 0), Node(cmap, 5))
        self.assertEqual(6, cmap.nb_faces())
        self.assertEqual(11, cmap.nb_nodes())
        self.assertTrue(Node(cmap, 10).get_dart().id > 0)
        plot_mesh(cmap)
        # Flip edge 0-8
        self.assertTrue(cmap.find_inner_edge(Node(cmap, 0), Node(cmap, 8))[0])
        flip_edge_cw(cmap, Node(cmap, 0), Node(cmap, 8))
        self.assertFalse(cmap.find_inner_edge(Node(cmap, 0), Node(cmap, 8))[0])
        plot_mesh(cmap)

        # Collapse node 8 (0.5, 0.5) from edge 8-10
        collapse_edge(cmap, Node(cmap, 8), Node(cmap, 10))
        self.assertEqual(5, cmap.nb_faces())
        self.assertEqual(10, cmap.nb_nodes())
        self.assertTrue(Node(cmap, 8).get_dart().id < 0)

        # Collapse node 10 (0.5, 0.25) from edge 10-5
        collapse_edge(cmap, Node(cmap, 10), Node(cmap, 5))
        self.assertEqual(4, cmap.nb_faces())
        self.assertEqual(9, cmap.nb_nodes())
        self.assertTrue(Node(cmap, 10).get_dart().id < 0)

        plot_mesh(cmap)

        self.assertEqual(global_score(cmap)[1], 0)



