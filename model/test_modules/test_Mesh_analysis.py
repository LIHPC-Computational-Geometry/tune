import unittest
import model.Linear2CMap as Linear2CMap
import model.Mesh_analysis as Mesh_analysis
import json


class TestMeshAnalysis(unittest.TestCase):

    def test_mesh_global_score(self):
        cmap = open_json_file("reg_mesh.json")
        nodes_score, mesh_score, mesh_ideal_score = Mesh_analysis.global_score(cmap)
        self.assertEqual((0,0), (mesh_score,mesh_ideal_score) )  # add assertion here


def open_json_file(file_name):
    f = open(file_name)
    json_mesh = json.load(f)
    return Linear2CMap.Mesh(json_mesh['nodes'], json_mesh['faces'])


if __name__ == '__main__':
    unittest.main()
