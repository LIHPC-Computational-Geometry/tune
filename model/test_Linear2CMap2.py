import unittest
import Linear2CMap

class TestLinear2CMap(unittest.TestCase):

    def test_triangle_json(self):
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

        self.assertEqual(json_cmap.nodes.all(), cmap.nodes.all())
        self.assertEqual(json_cmap.dart_info.all(), cmap.dart_info.all())

if __name__ == '__main__':
    unittest.main()
