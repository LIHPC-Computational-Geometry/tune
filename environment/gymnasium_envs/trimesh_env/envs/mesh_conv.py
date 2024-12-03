import numpy as np
from mesh_model.mesh_analysis import global_score, isValidAction, find_template_opposite_node
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_struct.mesh import Mesh

FLIP = 0

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
    nb_valid_darts = 0

    for d_info in mesh.dart_info:
        d_id = d_info[0]
        if isValidAction(mesh, d_id, FLIP):
            nb_valid_darts += 1
            dart_ids.append(d_id)
            d = Dart(mesh, d_id)
            A = d.get_node()
            d1 = d.get_beta(1)
            B = d1.get_node()
            d11 = d1.get_beta(1)
            C = d11.get_node()

            #Template niveau 1
            template[nb_valid_darts-1, 0] = nodes_scores[C.id]
            template[nb_valid_darts-1, 1] = nodes_scores[A.id]
            template[nb_valid_darts-1, 2] = nodes_scores[B.id]

            #template niveau 2

            n_id = find_template_opposite_node(d)
            if n_id is not None:
                template[nb_valid_darts-1, 3] = nodes_scores[n_id]
            n_id = find_template_opposite_node(d1)
            if n_id is not None:
                template[nb_valid_darts-1, 4] = nodes_scores[n_id]
            n_id = find_template_opposite_node(d11)
            if n_id is not None:
                template[nb_valid_darts-1, 5] = nodes_scores[n_id]

    valid_template = template[:nb_valid_darts-1, :]
    score_sum = np.sum(np.abs(valid_template), axis=1)
    indices_top_10 = np.argsort(score_sum)[-5:][::-1]
    valid_dart_ids = [dart_ids[i] for i in indices_top_10]
    X = valid_template[indices_top_10, :]
    X = X.flatten()
    return X, np.array(valid_dart_ids)