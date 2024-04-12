from model.mesh_struct.mesh import Mesh
from model.Mesh_analysis import global_score


class MeshDisplay:
    def __init__(self, m: Mesh):
        self.mesh = m

    def get_nodes_coordinates(self):
        """
        Build a list containing the coordinates of the all the mesh nodes
        :return: a list of coordinates (x,y)
        """
        node_list = []
        for n in self.mesh.nodes:
            node_list.append((n[0], n[1]))
        return node_list

    def get_edges(self):
        """
        Build a list containing the coordinates of the all the mesh nodes
        :return: a list of coordinates (x,y)
        """
        edge_list = []
        for d in self.mesh.dart_info:
            n1_id = d[3]
            n2_id = self.mesh.dart_info[d[1],3]
            if (d[2] != -1 and n1_id < n2_id) or d[2] == -1:
                edge_list.append((n1_id, n2_id))
        return edge_list

    def get_scores(self):
        """
        Calculates the irregularities of each node and the real and ideal score of the mesh
        :return: a list of three elements (nodes_score, mesh_score, ideal_mesh_score)
        """
        scores = global_score(self.mesh)
        return scores
