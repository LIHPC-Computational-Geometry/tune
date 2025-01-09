import numpy as np
from mesh_model.mesh_analysis import global_score, isValidAction, find_template_opposite_node
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_struct.mesh import Mesh


def get_x(state: Mesh, n_darts_selected: int, deep :int, degree: bool, restricted:bool, nodes_scores: list[int], nodes_adjacency: list[int]):
    mesh = state
    if degree:
        template, darts_id = get_template_deg(mesh, deep, nodes_scores, nodes_adjacency)
    else:
        template, darts_id = get_template(mesh, deep, nodes_scores)

    if restricted:
        darts_to_delete = []
        darts_id = []
        for i, d_info in enumerate(mesh.active_darts()):
            d_id = d_info[0]
            if d_info[2] == -1 or not isValidAction(mesh, d_info[0], 4)[0]:  # test the validity of all action type
                darts_to_delete.append(i)
            else:
                darts_id.append(d_id)
        valid_template = np.delete(template, darts_to_delete, axis=0)
    else:
        valid_template = template
    score_sum = np.sum(np.abs(valid_template[:,:deep]), axis=1)
    indices_selected_darts = np.argsort(score_sum)[-n_darts_selected:][::-1]
    valid_dart_ids = [darts_id[i] for i in indices_selected_darts]
    X = valid_template[indices_selected_darts, :]
    return X, np.array(valid_dart_ids)


def get_template(mesh: Mesh, deep: int, nodes_scores):
    size = len(mesh.dart_info)
    template = np.zeros((size, deep), dtype=np.int64)
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

        # Template niveau 1
        template[n_darts - 1, 0] = nodes_scores[C.id]
        template[n_darts - 1, 1] = nodes_scores[A.id]
        template[n_darts - 1, 2] = nodes_scores[B.id]

        if deep>3:
            # template niveau 2 deep = 6
            n_id = find_template_opposite_node(d)
            if n_id is not None:
                template[n_darts - 1, 3] = nodes_scores[n_id]
            n_id = find_template_opposite_node(d1)
            if n_id is not None:
                template[n_darts - 1, 4] = nodes_scores[n_id]
            n_id = find_template_opposite_node(d11)
            if n_id is not None:
                template[n_darts - 1, 5] = nodes_scores[n_id]

        if deep>6:
            # template niveau 3 - deep = 12
            d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)
            #Triangle F2
            n_id = find_template_opposite_node(d21)
            if n_id is not None:
                template[n_darts - 1, 6] = nodes_scores[n_id]
            n_id = find_template_opposite_node(d211)
            if n_id is not None:
                template[n_darts - 1, 7] = nodes_scores[n_id]
            # Triangle T3
            d12 = d1.get_beta(2)
            d121 = d12.get_beta(1)
            d1211 = d121.get_beta(1)
            n_id = find_template_opposite_node(d121)
            if n_id is not None:
                template[n_darts - 1, 8] = nodes_scores[n_id]
            n_id = find_template_opposite_node(d1211)
            if n_id is not None:
                template[n_darts - 1, 9] = nodes_scores[n_id]
            # Triangle T4
            d112 = d11.get_beta(2)
            d1121 = d112.get_beta(1)
            d11211 = d1121.get_beta(1)
            n_id = find_template_opposite_node(d1121)
            if n_id is not None:
                template[n_darts - 1, 10] = nodes_scores[n_id]
            n_id = find_template_opposite_node(d11211)
            if n_id is not None:
                template[n_darts - 1, 11] = nodes_scores[n_id]

    template = template[:n_darts, :]

    return template, dart_ids

def get_template_deg(mesh: Mesh, deep: int, nodes_scores, nodes_adjacency):
    size = len(mesh.dart_info)
    template = np.zeros((size, deep*2), dtype=np.int64)
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

        # Template niveau 1
        template[n_darts - 1, 0] = nodes_scores[C.id]
        template[n_darts - 1, deep] = nodes_adjacency[C.id]
        template[n_darts - 1, 1] = nodes_scores[A.id]
        template[n_darts - 1, deep+1] = nodes_adjacency[A.id]
        template[n_darts - 1, 2] = nodes_scores[B.id]
        template[n_darts - 1, deep+2] = nodes_adjacency[B.id]

        if deep>3:
            # template niveau 2
            n_id = find_template_opposite_node(d)
            if n_id is not None:
                template[n_darts - 1, 3] = nodes_scores[n_id]
                template[n_darts - 1, deep+3] = nodes_adjacency[n_id]
            n_id = find_template_opposite_node(d1)
            if n_id is not None:
                template[n_darts - 1, 4] = nodes_scores[n_id]
                template[n_darts - 1, deep+4] = nodes_adjacency[n_id]
            n_id = find_template_opposite_node(d11)
            if n_id is not None:
                template[n_darts - 1, 5] = nodes_scores[n_id]
                template[n_darts - 1, deep+5] = nodes_adjacency[n_id]

        if deep>6:
            # template niveau 3 - deep = 12
            if d.get_beta(2) is not None:
                d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)
                #Triangle F2
                n_id = find_template_opposite_node(d21)
                if n_id is not None:
                    template[n_darts - 1, 6] = nodes_scores[n_id]
                    template[n_darts - 1, deep+6] = nodes_adjacency[n_id]
                n_id = find_template_opposite_node(d211)
                if n_id is not None:
                    template[n_darts - 1, 7] = nodes_scores[n_id]
                    template[n_darts - 1, deep+7] = nodes_adjacency[n_id]
            # Triangle T3
            d12 = d1.get_beta(2)
            if d12 is not None:
                d121 = d12.get_beta(1)
                d1211 = d121.get_beta(1)
                n_id = find_template_opposite_node(d121)
                if n_id is not None:
                    template[n_darts - 1, 8] = nodes_scores[n_id]
                    template[n_darts - 1, deep+8] = nodes_adjacency[n_id]
                n_id = find_template_opposite_node(d1211)
                if n_id is not None:
                    template[n_darts - 1, 9] = nodes_scores[n_id]
                    template[n_darts - 1, deep+9] = nodes_adjacency[n_id]
            # Triangle T4
            d112 = d11.get_beta(2)
            if d112 is not None:
                d1121 = d112.get_beta(1)
                d11211 = d1121.get_beta(1)
                n_id = find_template_opposite_node(d1121)
                if n_id is not None:
                    template[n_darts - 1, 10] = nodes_scores[n_id]
                    template[n_darts - 1, deep+10] = nodes_adjacency[n_id]
                n_id = find_template_opposite_node(d11211)
                if n_id is not None:
                    template[n_darts - 1, 11] = nodes_scores[n_id]
                    template[n_darts - 1, deep+11] = nodes_adjacency[n_id]

    template = template[:n_darts, :]
    return template, dart_ids