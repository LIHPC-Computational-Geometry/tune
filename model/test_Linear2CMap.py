import unittest
import Linear2CMap

class TestLinear2CMap(unittest.TestCase):
    def test_nodes(self):
        cmap = Linear2CMap.Mesh()
        self.assertEqual(0, cmap.nb_vertices())
        n = cmap.add_vertex(1.1,2.3)
        self.assertEqual(1, cmap.nb_vertices())
        self.assertEqual(1.1, n.x())
        self.assertEqual(2.3, n.y())
        self.assertEqual(1.1, n.xy()[0])
        self.assertEqual(2.3, n.xy()[1])
        n.set_x(3)
        self.assertEqual(3, n.x())
        n2 = cmap.add_vertex(1, 23)
        n3 = cmap.add_vertex(3, 1)
        self.assertEqual(3, cmap.nb_vertices())
        cmap.del_vertex(n2.id)
        self.assertEqual(2, cmap.nb_vertices())
        cmap.del_vertex(n.id)
        self.assertEqual(1, cmap.nb_vertices())
        cmap.del_vertex(n3.id)
        self.assertEqual(0, cmap.nb_vertices())

    def test_triangles(self):
        cmap = Linear2CMap.Mesh()
        n1 = cmap.add_vertex(0,0)
        n2 = cmap.add_vertex(0,1)
        n3 = cmap.add_vertex(1,0)
        cmap.add_triangle(n1,n2,n3)
        self.assertEqual(1, cmap.nb_faces())

if __name__ == '__main__':
    unittest.main()
