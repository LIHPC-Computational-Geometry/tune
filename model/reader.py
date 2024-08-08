import string

from model.mesh_struct.mesh import Mesh


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
            for n in range(int(nb_vertices)):
                line = f.readline()
                ls = line.split()
                x = float(ls[0])
                y = float(ls[1])
                nodes.append([x, y])
        if "Triangles" == line.strip():
            line = f.readline()
            nb_faces = line.strip()
            for e in range(int(nb_faces)):
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
            for b in range(nb_blocs):
                line = f.readline()
                ls = line.split()
                nb_nodes_b = int(ls[3])
                # skip the tags
                for n in range(nb_nodes_b):
                    line = f.readline()
                for n in range(nb_nodes_b):
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
            for b in range(nb_blocs):
                line = f.readline()
                ls = line.split()
                elem_type = int(ls[2])
                nb_elems_b = int(ls[3])
                for e in range(nb_elems_b):
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
                    else:
                        print("element_type " + str(elem_type) + " not handled")
                        exit(1)

    f.close()
    mesh = Mesh(nodes, faces)

    return mesh
