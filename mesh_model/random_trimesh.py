from __future__ import annotations
import string
import os
import numpy as np

from mesh_model.mesh_analysis.global_mesh_analysis import NodeAnalysis
from mesh_model.mesh_struct.mesh_elements import Dart, Node
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.trimesh_analysis import TriMeshQualityAnalysis
from environment.actions.triangular_actions import flip_edge_ids, split_edge_ids, collapse_edge_ids
from mesh_model.writer import write_json


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
    m_analysis = TriMeshQualityAnalysis(mesh)

    num_nodes = 3
    dart_id = 0

    while dart_id < len(mesh.dart_info):
        d_info = mesh.dart_info[dart_id]
        d = Dart(mesh, d_info[0])
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        x_C, y_C = m_analysis.find_opposite_node(d)

        # Search if the node C already exist in the actual mesh
        found, n_id = m_analysis.node_in_mesh( x_C, y_C)

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
    m_analysis.set_adjacency()
    m_analysis.set_scores()
    m_analysis.set_geometric_quality()
    m_analysis.set_is_starred()
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
    mesh_shuffle(mesh, num_nodes_max)
    return mesh


def mesh_shuffle_flip(mesh: Mesh) -> Mesh:
    """
    Performs random flip actions on mesh darts.
    :param mesh: the mesh to work with
    :return: a mesh with randomly flipped darts.
    """
    nb_flip = len(mesh.dart_info)
    nb_nodes = len(mesh.nodes)
    m_analysis = TriMeshQualityAnalysis(mesh)
    for i in range(nb_flip):
        i1 = np.random.randint(nb_nodes)
        i2 = np.random.randint(nb_nodes)
        if i1 != i2:
            flip_edge_ids(m_analysis, i1, i2)
    return mesh

def mesh_shuffle(mesh: Mesh, num_nodes) -> Mesh:
    """
    Performs random actions on mesh darts.
    :param mesh: the mesh to work with
    :param num_nodes: number nodes of the mesh
    :return: a mesh with randomly flipped darts.
    """
    nb_action_max = 10
    nb_action = 0
    active_darts_list = mesh.active_darts()
    m_analysis = TriMeshQualityAnalysis(mesh)
    i = 0
    while nb_action < nb_action_max:
        action_type = np.random.randint(0, 3)
        d_id = np.random.randint(len(active_darts_list))
        d_id = active_darts_list[d_id][0]
        dart = Dart(mesh, d_id)
        i1 = dart.get_node()
        i2 = ((dart.get_beta(1)).get_beta(1)).get_node()
        if action_type == 0 and m_analysis.isValidAction(d_id, action_type)[0]:
            flip_edge_ids(m_analysis, i1.id, i2.id)
            nb_action += 1
        elif action_type == 1 and m_analysis.isValidAction(d_id, action_type)[0]:
            split_edge_ids(m_analysis, i1.id, i2.id)
            nb_action += 1
        elif action_type == 2 and m_analysis.isValidAction(d_id, action_type)[0]:
            collapse_edge_ids(m_analysis, i1.id, i2.id)
            nb_action += 1
        i += 1
        active_darts_list = mesh.active_darts()
    return mesh

def random_dataset(nb_nodes: int, mesh_dir: string) -> None:
    """
    Save random meshes in destination folder.
    :param nb_nodes: number of nodes of the initial regular mesh we generate
    :param nb_rd_actions: number of actions performed on the mesh to deteriorate
    :param mesh_dir: directory to save meshes in json format
    """
    nb_meshes = 100
    for i in range(nb_meshes):
        mesh = random_mesh(nb_nodes)
        if i < 10:
            filename = "mesh_00"+str(i)+".json"
        elif i<100:
            filename = "mesh_0"+str(i)+".json"
        else:
            filename = "mesh_"+str(i)+".json"
        filepath = os.path.join(mesh_dir, filename)
        write_json(filepath, mesh)

if __name__ == '__main__':
    random_dataset(30,"../training/dataset/test_random_30")