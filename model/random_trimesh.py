from __future__ import annotations
import numpy as np

from model.mesh_struct.mesh_elements import Dart, Node
from model.mesh_struct.mesh import Mesh
from model.mesh_analysis import find_opposite_node, node_in_mesh
from actions.triangular_actions import flip_edge_ids, split_edge_ids


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
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
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


def random_flip_mesh(num_nodes_max: int) -> Mesh:
    """
    Create a random mesh with a fixed number of nodes.
    :param num_nodes_max: number of nodes of the final mesh
    :return: a random mesh
    """
    mesh = regular_mesh(num_nodes_max)
    mesh_shuffle_flip(mesh)
    return mesh

def random_mesh(num_nodes_max: int) -> Mesh:
    """
    Create a random mesh with a fixed number of nodes.
    :param num_nodes_max: number of nodes of the final mesh
    :return: a random mesh
    """
    mesh = regular_mesh(num_nodes_max)
    mesh_shuffle_flip(mesh)
    return mesh


def mesh_shuffle_flip(mesh: Mesh) -> Mesh:
    """
    Performs random flip actions on mesh darts.
    :param mesh: the mesh to work with
    :return: a mesh with randomly flipped darts.
    """
    nb_flip = len(mesh.dart_info)*2
    nb_nodes = len(mesh.nodes)
    for i in range(nb_flip):
        i1 = np.random.randint(nb_nodes)
        i2 = np.random.randint(nb_nodes)
        if i1 != i2:
            flip_edge_ids(mesh, i1, i2)
    return mesh

def mesh_shuffle(mesh: Mesh) -> Mesh:
    """
    Performs random flip actions on mesh darts.
    :param mesh: the mesh to work with
    :return: a mesh with randomly flipped darts.
    """
    nb_flip = len(mesh.dart_info)
    nb_action =nb_flip * 2
    nb_nodes = len(mesh.nodes)
    for i in range(nb_action):
        i1 = np.random.randint(nb_nodes)
        i2 = np.random.randint(nb_nodes)
        if i1 != i2 and i%2 == 0:
            flip_edge_ids(mesh, i1, i2)
        elif i1 !=i2 :
            split_edge_ids(mesh, i1, i2)
    return mesh




