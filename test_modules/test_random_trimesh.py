import unittest
from model.random_trimesh import regular_mesh, random_mesh


class TestRandomTrimesh(unittest.TestCase):

    def test_regular_trimesh(self):
        m = regular_mesh(44)
        self.assertEqual(m.nb_nodes(), 44)
        m = regular_mesh(30)
        self.assertEqual(m.nb_nodes(), 30)
        m = regular_mesh(60)
        self.assertEqual(m.nb_nodes(), 60)

    def test_random_trimesh(self):
        m = random_mesh(44)
        self.assertEqual(m.nb_nodes(), 44)
        m = random_mesh(30)
        self.assertEqual(m.nb_nodes(), 30)
        m = random_mesh(60)
        self.assertEqual(m.nb_nodes(), 60)



if __name__ == '__main__':
    unittest.main()
