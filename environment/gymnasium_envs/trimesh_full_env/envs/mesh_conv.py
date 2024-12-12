import numpy as np
from mesh_model.mesh_analysis import global_score, isValidAction, find_template_opposite_node
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_struct.mesh import Mesh


def get_x_level_2(state: Mesh):
    """
    Get the feature vector of the state.The vector is obtained by convolution operations on every dart. The resulting matrix represent the neighbour of the 10 darts with more irregularities in his neighbiour.
    :param state: the state
    :return: the feature vector
    """
    mesh = state
    nodes_scores = global_score(mesh)[0]
    size = len(mesh.dart_info)
    template = np.zeros((size, 6), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()

        #Template niveau 1
        template[n_darts-1, 0] = nodes_scores[C.id]
        template[n_darts-1, 1] = nodes_scores[A.id]
        template[n_darts-1, 2] = nodes_scores[B.id]

        #template niveau 2

        n_id = find_template_opposite_node(d)
        if n_id is not None:
            template[n_darts-1, 3] = nodes_scores[n_id]
        n_id = find_template_opposite_node(d1)
        if n_id is not None:
            template[n_darts-1, 4] = nodes_scores[n_id]
        n_id = find_template_opposite_node(d11)
        if n_id is not None:
            template[n_darts-1, 5] = nodes_scores[n_id]

    valid_template = template[:n_darts-1, :]
    score_sum = np.sum(np.abs(valid_template), axis=1)
    indices_top_20 = np.argsort(score_sum)[-20:][::-1]
    valid_dart_ids = [dart_ids[i] for i in indices_top_20]
    X = valid_template[indices_top_20, :]
    X = X.flatten()
    return X, np.array(valid_dart_ids)


def get_x_level_2_deg(state: Mesh):
    """
    Get the feature vector of the state.The vector is obtained by convolution operations on every dart. The resulting matrix represent the neighbour of the 10 darts with more irregularities in his neighbiour.
    :param state: the state
    :return: the feature vector
    """
    mesh = state
    nodes_scores, _, _, nodes_adjacency = global_score(mesh)
    size = len(mesh.dart_info)
    template = np.zeros((size, 6*2), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()

        #Template niveau 1
        template[n_darts-1, 0] = nodes_scores[C.id]
        template[n_darts - 1, 6] = nodes_adjacency[C.id]
        template[n_darts-1, 1] = nodes_scores[A.id]
        template[n_darts - 1, 7] = nodes_adjacency[A.id]
        template[n_darts-1, 2] = nodes_scores[B.id]
        template[n_darts - 1, 8] = nodes_adjacency[B.id]
        #template niveau 2

        n_id = find_template_opposite_node(d)
        if n_id is not None:
            template[n_darts-1, 3] = nodes_scores[n_id]
            template[n_darts-1, 9] = nodes_adjacency[n_id]
        n_id = find_template_opposite_node(d1)
        if n_id is not None:
            template[n_darts-1, 4] = nodes_scores[n_id]
            template[n_darts-1, 10] = nodes_adjacency[n_id]
        n_id = find_template_opposite_node(d11)
        if n_id is not None:
            template[n_darts-1, 5] = nodes_scores[n_id]
            template[n_darts-1, 11] = nodes_adjacency[n_id]

    valid_template = template[:n_darts-1, :]
    score_sum = np.sum(np.abs(valid_template), axis=1)
    indices_top_20 = np.argsort(score_sum)[-20:][::-1]
    valid_dart_ids = [dart_ids[i] for i in indices_top_20]
    X = valid_template[indices_top_20, :]
    return X, np.array(valid_dart_ids)