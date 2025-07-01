from __future__ import annotations
import numpy as np
import os

from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.quadmesh_analysis import QuadMeshOldAnalysis
from environment.actions.quadrangular_actions import flip_edge_cntcw_ids, split_edge_ids, collapse_edge_ids
from mesh_model.reader import read_gmsh


def random_mesh() -> Mesh:
    """
    Create a random mesh with a fixed number of nodes.
    :param num_nodes_max: number of nodes of the final mesh
    :return: a random mesh
    """
    filename = os.path.join(os.path.dirname(__file__), '../mesh_files/simple_quad.msh')
    #filename = os.path.join('../mesh_files/', 't1_quad.msh')
    #mesh = read_gmsh("/home/ropercha/PycharmProjects/tune/mesh_files/t1_quad.msh")
    mesh = read_gmsh(filename)
    mesh_shuffle(mesh, 5)
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
    m_analysis = QuadMeshOldAnalysis(mesh)
    i = 0
    while i < nb_action_max:
        action_type = np.random.randint(0, 3)
        d_id = np.random.randint(len(active_darts_list))
        d_id = active_darts_list[d_id][0]
        dart = Dart(mesh, d_id)
        i1 = dart.get_node()
        i2 = (dart.get_beta(1)).get_node()
        #plot_mesh(mesh)
        if action_type == 0 and m_analysis.isValidAction(d_id, action_type)[0]:
            flip_edge_cntcw_ids(m_analysis, i1.id, i2.id)
            nb_action += 1
        elif action_type == 1 and m_analysis.isValidAction(d_id, action_type)[0]:
            split_edge_ids(m_analysis, i1.id, i2.id)
            nb_action += 1
        elif action_type == 2 and m_analysis.isValidAction(d_id, action_type)[0]:
            collapse_edge_ids(m_analysis, i1.id, i2.id)
            nb_action += 1
        elif action_type == 3 and m_analysis.isValidAction(d_id, action_type)[0]:
            collapse_edge_ids(m_analysis, i1.id, i2.id)
            nb_action += 1
        i += 1
        active_darts_list = mesh.active_darts()
    return mesh
