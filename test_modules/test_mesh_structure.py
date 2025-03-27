import unittest
import mesh_model.mesh_struct.mesh as mesh
import numpy.testing


class TestMeshStructure(unittest.TestCase):

    def test_empty_mesh(self):
        cmap = mesh.Mesh()
        self.assertEqual(0, cmap.nb_nodes())
        self.assertEqual(0, cmap.nb_faces())

    def test_nodes(self):
        cmap = mesh.Mesh()
        self.assertEqual(0, cmap.nb_nodes())

        n = cmap.add_node(1.1, 2.3)
        self.assertEqual(1.1, n.x())
        self.assertEqual(2.3, n.y())
        n.set_x(3)
        self.assertEqual(3, n.x())
        n.set_y(4)
        self.assertEqual(4, n.y())
        n.set_xy(5, 6)
        self.assertEqual(5, n.x())
        self.assertEqual(6, n.y())
        n2 = cmap.add_node(1, 23)
        n3 = cmap.add_node(3, 1)
        cmap.add_triangle(n, n2, n3)
        self.assertEqual(3, cmap.nb_nodes())
        cmap.del_vertex(n2)
        self.assertEqual(2, cmap.nb_nodes())
        cmap.del_vertex(n)
        self.assertEqual(1, cmap.nb_nodes())
        cmap.del_vertex(n3)
        self.assertEqual(0, cmap.nb_nodes())

    def test_single_triangle(self):
        cmap = mesh.Mesh()
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
        json_cmap = mesh.Mesh(json_mesh['nodes'], json_mesh['faces'])

        # same mesh programmatically
        cmap = mesh.Mesh()
        n0 = cmap.add_node(0, 1)
        n1 = cmap.add_node(1, 1)
        n2 = cmap.add_node(1, 0)
        cmap.add_triangle(n2, n0, n1)

        numpy.testing.assert_array_equal(json_cmap.nodes, cmap.nodes)
        numpy.testing.assert_array_equal(json_cmap.dart_info, cmap.dart_info)

    def test_single_quad(self):
        cmap = mesh.Mesh()
        n1 = cmap.add_node(0, 0)
        n2 = cmap.add_node(1, 0)
        n3 = cmap.add_node(1, 1)
        n4 = cmap.add_node(0, 1)
        t = cmap.add_quad(n1, n2, n3, n4)

        nodes_of_t = t.get_nodes()
        self.assertEqual(1, cmap.nb_faces())
        self.assertEqual(4, len(nodes_of_t))
        self.assertEqual(n1, nodes_of_t[0])
        self.assertEqual(n2, nodes_of_t[1])
        self.assertEqual(n3, nodes_of_t[2])
        self.assertEqual(n4, nodes_of_t[3])


if __name__ == '__main__':
    unittest.main()
