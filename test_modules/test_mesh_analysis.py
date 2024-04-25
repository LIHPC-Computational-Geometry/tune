import unittest
from model.mesh_struct.mesh import Mesh
import model.mesh_analysis as Mesh_analysis


class TestMeshAnalysis(unittest.TestCase):

    def test_mesh_regular_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0,0.0],[-0.5,-1.0],[0.0,-2.0], [0.5,-1.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 5, 0], [0, 5, 6], [0, 6, 8], [6, 7, 8], [0, 8, 1]]
        cmap = Mesh(nodes,faces)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((0,0), (mesh_score,mesh_ideal_score) )  # add assertion here

    def test_mesh_with_irregularities(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0, 2.0], [-2.0, 1.0], [-1.0, 0.0],
                 [-2.0, 0.0], [-2.0, -1.0], [-0.5, -1.0], [-1.0, -2.0], [0.0, -2.0], [1.0, -2.0],
                 [0.5, -1.0], [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [1.0, 2.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [7, 0, 3], [7, 10, 0], [10, 14, 0], [0, 14, 1], [10, 12, 14],
                 [3, 4, 5], [3, 5, 6], [3, 6, 7], [7, 6, 8], [7, 8, 9], [7, 9, 10], [10, 9, 11], [10, 11, 12],
                 [14, 12, 13], [14, 13, 15], [1, 14, 15], [1, 15, 16], [1, 16, 17], [1, 17, 2], [2, 17, 18], [2, 18, 4]]
        cmap = Mesh(nodes,faces)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((6, -2), (mesh_score,mesh_ideal_score) )  # add assertion here
        

    def test_mesh_bad_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((3, 1), (mesh_score, mesh_ideal_score))

    def test_split_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0]]
        faces = [[0, 1, 2], [0, 2, 3], [1, 4, 2]]
        cmap = Mesh(nodes, faces)
        cmap.split_edge_ids(0,2)
        cmap.split_edge_ids(1, 2)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((5, 1), (mesh_score, mesh_ideal_score))


if __name__ == '__main__':
    unittest.main()
