import numpy as np
from actions.triangular_actions import flip_edge_ids
from model.mesh_struct.mesh_elements import Face, Dart, Node
from model.mesh_analysis import find_opposite_node, node_in_mesh, isValidAction

FLIP = 0

GLOBAL = 0


def get_x(state, a, env, feat):
    """
    Get the feature vector of the state-action pair
    :param state: the state
    :param a: the action
    :return: the feature vector
    """
    if feat == GLOBAL:
        return get_x_global_4(env)


def get_x_global_4(env):
    """
    Get the feature vector of the state-action pair
    :param env: The environment
    :return: the feature vector
    """

    mesh = env.mesh
    size = env.size
    template = np.zeros((size, 6))

    for d_info in mesh.dart_info:

        d = Dart(mesh, d_info[0])
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()

        #Template niveau 1
        template[d_info[0], 0] = env.nodes_scores[C.id]
        template[d_info[0], 1] = env.nodes_scores[A.id]
        template[d_info[0], 2] = env.nodes_scores[B.id]

        #template niveau 2

        x_D, y_D = find_opposite_node(d)
        found, n_id = node_in_mesh(mesh, x_D, y_D)
        if found:
            template[d_info[0], 3] = env.nodes_scores[n_id]
        x_E, y_E = find_opposite_node(d1)
        found, n_id = node_in_mesh(mesh, x_E, y_E)
        if found:
            template[d_info[0], 4] = env.nodes_scores[n_id]
        x_F, y_F = find_opposite_node(d11)
        found, n_id = node_in_mesh(mesh, x_F, y_F)
        if found:
            template[d_info[0], 5] = env.nodes_scores[n_id]

    dart_to_delete = []
    dart_ids = []
    for i in range(size):
        d = Dart(mesh, i)
        if not isValidAction(mesh, d.id):
            dart_to_delete.append(i)
        else :
            dart_ids.append(i)
    valid_template = np.delete(template, dart_to_delete, axis=0)
    score_sum = np.sum(np.abs(valid_template), axis=1)
    indices_top_10 = np.argsort(score_sum)[-10:][::-1]
    valid_dart_ids = [dart_ids[i] for i in indices_top_10]
    X = valid_template[indices_top_10, :]
    X = X.flatten()
    return X, valid_dart_ids

    """
    if X.std() != 0.0:
        X = (X - X.mean()) / X.std()
    return X, indices_top_10"""

"""
def get_x_partial(state, a, env):
    X=np.zeros(13)
    X[:9] = get_x_local(state, a, env)
    X[9:] = get_closest_goal_direction(state, env)
    return X

"""