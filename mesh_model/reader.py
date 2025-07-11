import string
import json
import os

from mesh_model.mesh_struct.mesh import Mesh

def getMeshFiles(d):
    l = []
    for sdp, Lsd, Lnf in os.walk(d):
        for f in Lnf:
            if (f.endswith(".msh")):
                l.append(f)
    l.sort()
    return l

def getFiles(d):
    l = []
    for sdp, Lsd, Lnf in os.walk(d):
        for f in Lnf:
            l.append(f)
    l.sort()
    return l

def read_medit(filename: string) -> Mesh:
    """
    Create a mesh read from a Medit .mesh file.
    :param filename: name of the file
    :return: a mesh
    """

    f = open(filename, 'r')

    nodes = []
    faces = []
    while True:
        line = f.readline()
        if not line:
            break
        if "Vertices" == line.strip():
            line = f.readline()
            nb_vertices = line.strip()
            for _ in range(int(nb_vertices)):
                line = f.readline()
                ls = line.split()
                x = float(ls[0])
                y = float(ls[1])
                nodes.append([x, y])
        if "Triangles" == line.strip():
            line = f.readline()
            nb_faces = line.strip()
            for _ in range(int(nb_faces)):
                line = f.readline()
                ls = line.split()
                n0 = int(ls[0])
                n1 = int(ls[1])
                n2 = int(ls[2])
                faces.append([n0 - 1, n1 - 1, n2 - 1])

    f.close()
    mesh = Mesh(nodes, faces)

    return mesh


def read_gmsh(filename: string) -> Mesh:
    """
    Create a mesh read from a gmsh .msh file.
    :param filename: name of the file
    :return: a mesh
    """

    f = open(filename, 'r')

    nodes = []
    faces = []
    while True:
        line = f.readline()
        if not line:
            break
        if "$Nodes" == line.strip():
            line = f.readline()
            ls = line.split()
            nb_blocs = int(ls[0])
            for _ in range(nb_blocs):
                line = f.readline()
                ls = line.split()
                nb_nodes_b = int(ls[3])
                # skip the tags
                for _ in range(nb_nodes_b):
                    line = f.readline()
                for _ in range(nb_nodes_b):
                    line = f.readline()
                    ls = line.split()
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])
                    nodes.append([x, y, z])

        if "$Elements" == line.strip():
            line = f.readline()
            ls = line.split()
            nb_blocs = int(ls[0])
            for _ in range(nb_blocs):
                line = f.readline()
                ls = line.split()
                elem_type = int(ls[2])
                nb_elems_b = int(ls[3])
                for _ in range(nb_elems_b):
                    line = f.readline()
                    ls = line.split()
                    if elem_type == 2:
                        n0 = int(ls[1])
                        n1 = int(ls[2])
                        n2 = int(ls[3])
                        faces.append([n0 - 1, n1 - 1, n2 - 1])
                    elif elem_type == 3:
                        n0 = int(ls[1])
                        n1 = int(ls[2])
                        n2 = int(ls[3])
                        n3 = int(ls[4])
                        faces.append([n0 - 1, n1 - 1, n2 - 1, n3 - 1])
                    elif elem_type == 1:  # skip 2-node line elements
                        continue
                    elif elem_type == 15: # skip 1-node point elements
                        continue
                    else:
                        print("element_type " + str(elem_type) + " not handled")
                        exit(1)
    f.close()
    mesh = Mesh(nodes, faces)

    return mesh

def read_json(filename) -> Mesh:
    with open(filename, 'r') as f:
        json_mesh = json.load(f)
    mesh = Mesh(json_mesh['nodes'], json_mesh['faces'])
    return mesh

def read_dataset(dataset_dir) -> list[Mesh]:
    mesh_dataset = []
    for f in getFiles(dataset_dir):
        if (f.endswith(".msh")):
            msh_f = dataset_dir + "/" + f
            cmap = read_gmsh(msh_f)
            mesh_dataset.append(cmap)
        elif (f.endswith(".json")):
            json_f = dataset_dir + "/" + f
            cmap = read_json(json_f)
            mesh_dataset.append(cmap)
    return mesh_dataset