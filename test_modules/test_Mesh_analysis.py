import unittest
from model.mesh_struct.mesh import Mesh
import model.Mesh_analysis as Mesh_analysis
import json


class TestMeshAnalysis(unittest.TestCase):

    def test_mesh_global_score(self):
        nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 1], [-0.5, 1.0], [0.0, 2.0], [-1.0,0.0],[-0.5,-1.0],[0.0,-2.0], [0.5,-1.0]]
        faces = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3 ,5 ,0], [0, 5, 6], [0, 6, 8], [6, 7, 8], [0, 8, 1]]
        cmap = Mesh(nodes,faces)
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((0,0), (mesh_score,mesh_ideal_score) )  # add assertion here


def open_json_file(file_name):
    f = open(file_name)
    json_mesh = json.load(f)
    return Mesh(json_mesh['nodes'], json_mesh['faces'])


if __name__ == '__main__':
    unittest.main()
