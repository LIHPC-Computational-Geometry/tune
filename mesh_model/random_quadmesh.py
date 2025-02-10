from __future__ import annotations
import numpy as np
import os

from mesh_model.mesh_struct.mesh_elements import Dart, Node
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.quadmesh_analysis import isValidAction
from actions.quadrangular_actions import flip_edge_ids, split_edge_ids, collapse_edge_ids, cleanup_edge_ids
from mesh_model.reader import read_gmsh


def random_mesh() -> Mesh:
    """
    Create a random mesh with a fixed number of nodes.
    :param num_nodes_max: number of nodes of the final mesh
    :return: a random mesh
    """
    filename = os.path.join('../mesh_files/', 't1_quad.msh')
    mesh = read_gmsh(filename)
    mesh_shuffle(mesh, 10)
    return mesh

def mesh_shuffle(mesh: Mesh, num_nodes) -> Mesh:
    """
    Performs random actions on mesh darts.
    :param mesh: the mesh to work with
    :param num_nodes: number nodes of the mesh
    :return: a mesh with randomly flipped darts.
    """
    nb_action_max = int(num_nodes)
    nb_action = 0
    active_darts_list = mesh.active_darts()
    i = 0
    while i < nb_action_max:
        action_type = np.random.randint(0, 3)
        d_id = np.random.randint(len(active_darts_list))
        d_id = active_darts_list[d_id][0]
        dart = Dart(mesh, d_id)
        i1 = dart.get_node()
        i2 = (dart.get_beta(1)).get_node()
        if action_type == 0 and isValidAction(mesh, d_id, action_type)[0]:
            flip_edge_ids(mesh, i1.id, i2.id)
            nb_action += 1
        elif action_type == 1: # and isValidAction(mesh, d_id, action_type)[0]
            split_edge_ids(mesh, i1.id, i2.id)
            nb_action += 1
        elif action_type == 2 and isValidAction(mesh, d_id, action_type)[0]:
            collapse_edge_ids(mesh, i1.id, i2.id)
            nb_action += 1
        elif action_type == 3 and isValidAction(mesh, d_id, action_type)[0]:
            cleanup_edge_ids(mesh, i1.id, i2.id)
            nb_action += 1
        i += 1
        active_darts_list = mesh.active_darts()
    return mesh
