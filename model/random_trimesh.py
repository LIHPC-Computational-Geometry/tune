from __future__ import annotations
import numpy as np

from model.mesh_struct.mesh_elements import Dart, Node
from model.mesh_struct.mesh import Mesh
from model.mesh_analysis import find_opposite_node
from actions.triangular_actions import flip_edge_ids


def regular_mesh(num_nodes_max: int) -> Mesh:
    """
    Create an almost regular mesh. Depending on the number of nodes, the mesh may contain irregularities on the boundaries.
    It starts with a fixed initial triangle, then creates neighboring triangles by traversing all darts of the mesh.
    :param num_nodes_max: number of nodes of the final mesh
    :return: an almost regular mesh
    """

    nodes = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]]
    faces = [[0, 1, 2]]
    mesh = Mesh(nodes, faces)

    num_nodes = 3
    dart_id = 0

    while dart_id < len(mesh.dart_info):
        d_info = mesh.dart_info[dart_id]
        d = Dart(mesh, d_info[0])
        x_C, y_C = find_opposite_node(d)

        # Search if the node C already exist in the actual mesh
        found, n_id = node_in_mesh(mesh, x_C, y_C)

        if found and d.get_beta(2) is None:
            C = Node(mesh, n_id)
            mesh.add_triangle(B, A, C)
            mesh.set_twin_pointers()
        elif not found and num_nodes < num_nodes_max:
            C = mesh.add_node(x_C, y_C)
            mesh.add_triangle(B, A, C)
            mesh.set_twin_pointers()
            num_nodes += 1

        dart_id += 1

    mesh.set_twin_pointers()

    return mesh


def random_mesh(num_nodes_max: int) -> Mesh:
    """
    Create a random mesh with a fixed number of nodes.
    :param num_nodes_max: number of nodes of the final mesh
    :return: a random mesh
    """
    mesh = regular_mesh(num_nodes_max)
    mesh_shuffle(mesh)
    return mesh


def mesh_shuffle(mesh: Mesh) -> Mesh:
    """
    Performs random flip actions on mesh darts.
    :param mesh: the mesh to work with
    :return: a mesh with randomly flipped darts.
    """
    nb_flip = len(mesh.dart_info)
    nb_nodes = len(mesh.nodes)
    for i in range(nb_flip):
        i1 = np.random.randint(nb_nodes)
        i2 = np.random.randint(nb_nodes)
        if i1 != i2:
            flip_edge_ids(mesh, i1, i2)
    return mesh


def node_in_mesh(mesh: Mesh, x: float, y: float) -> (bool, int):
    """
    Search if the node of coordinate (x, y) is inside the mesh.
    :param mesh: the mesh to work with
    :param x: X coordinate
    :param y: Y coordinate
    :return: a boolean indicating if the node is inside the mesh and the id of the node if it is.
    """
    n_id = 0
    for n in mesh.nodes:
        if abs(x - n[0]) <= 0.1 and abs(y - n[1]) <= 0.1:
            return True, n_id
        n_id = n_id + 1
    return False, None




