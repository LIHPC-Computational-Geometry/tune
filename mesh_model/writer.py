import string
import json
import os

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.reader import read_gmsh


def write_json(filename: string, mesh: Mesh) -> None:
    """
    Create a json file from a mesh structure.
    :param filename: name of the file
    :param mesh: mesh structure
    """

    #Get faces nodes
    nodes = [[float(list[0]), float(list[1])] for list in mesh.nodes]
    faces = []

    for d_id in mesh.active_faces():
        d = Dart(mesh, d_id)
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d111 = d11.get_beta(1)

        n1 = d.get_node()
        n2 = d1.get_node()
        n3 = d11.get_node()
        n4 = d111.get_node()

        if d111 == d:
            faces.append([int(n1.id), int(n2.id), int(n3.id)])
        else:
            faces.append([int(n1.id), int(n2.id), int(n3.id), int(n4.id)])


    data = {
        "nodes": nodes,
        "faces": faces
    }

    with open(filename, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    cmap = read_gmsh("../mesh_files/t1_tri.msh")
    write_json("t1_tri-test.json", cmap)