import unittest
import Linear2CMap
import numpy.testing


class TestLinear2CMap(unittest.TestCase):

    def test_empty_mesh(self):
        cmap = Linear2CMap.Mesh()
        self.assertEqual(0, cmap.nb_nodes())
        self.assertEqual(0, cmap.nb_faces())

    def test_nodes(self):
        cmap = Linear2CMap.Mesh()
        self.assertEqual(0, cmap.nb_nodes())

        n = cmap.add_node(1.1, 2.3)
        self.assertEqual(1, cmap.nb_nodes())
        self.assertEqual(1.1, n.x())
        self.assertEqual(2.3, n.y())
        n.set_x(3)
        self.assertEqual(3, n.x())
        n2 = cmap.add_node(1, 23)
        n3 = cmap.add_node(3, 1)
        self.assertEqual(3, cmap.nb_nodes())
        cmap.del_vertex(n2)
        self.assertEqual(2, cmap.nb_nodes())
        cmap.del_vertex(n)
        self.assertEqual(1, cmap.nb_nodes())
        cmap.del_vertex(n3)
        self.assertEqual(0, cmap.nb_nodes())

    def test_single_triangle(self):
        cmap = Linear2CMap.Mesh()
        n1 = cmap.add_node(0, 0)
        n2 = cmap.add_node(0, 1)
        n3 = cmap.add_node(1, 0)
        t = cmap.add_triangle(n1, n2, n3)

        nodes_of_t = t.get_nodes()
        self.assertEqual(1, cmap.nb_faces())
        self.assertEqual(3, len(nodes_of_t))
        self.assertEqual(n1, nodes_of_t[0])
        self.assertEqual(n2, nodes_of_t[1])
        self.assertEqual(n3, nodes_of_t[2])

    def test_json(self):
        # mesh with one triangle in json
        json_mesh = {
            "nodes": [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            "faces": [[2, 0, 1]]
        }
        json_cmap = Linear2CMap.Mesh(json_mesh['nodes'], json_mesh['faces'])

        # same mesh programmatically
        cmap = Linear2CMap.Mesh()
        n0 = cmap.add_node(0, 1)
        n1 = cmap.add_node(1, 1)
        n2 = cmap.add_node(1, 0)
        cmap.add_triangle(n2, n0, n1)

        numpy.testing.assert_array_equal(json_cmap.nodes, cmap.nodes)
        numpy.testing.assert_array_equal(json_cmap.dart_info, cmap.dart_info)

    def test_single_quad(self):
        cmap = Linear2CMap.Mesh()
        n1 = cmap.add_node(0, 0)
        n2 = cmap.add_node(0, 1)
        n3 = cmap.add_node(1, 1)
        n4 = cmap.add_node(1, 0)
        t = cmap.add_quad(n1, n2, n3, n4)

        nodes_of_t = t.get_nodes()
        self.assertEqual(1, cmap.nb_faces())
        self.assertEqual(4, len(nodes_of_t))
        self.assertEqual(n1, nodes_of_t[0])
        self.assertEqual(n2, nodes_of_t[1])
        self.assertEqual(n3, nodes_of_t[2])
        self.assertEqual(n4, nodes_of_t[3])

    def test_flip(self):
        cmap = Linear2CMap.Mesh()
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

        cmap.flip_edge(n00, n11)
        self.assertEqual(2, cmap.nb_faces())
        self.assertEqual(4, cmap.nb_nodes())

    def test_split(self):
        cmap = Linear2CMap.Mesh()
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

        cmap.split_edge(n00, n11)

if __name__ == '__main__':
    unittest.main()
