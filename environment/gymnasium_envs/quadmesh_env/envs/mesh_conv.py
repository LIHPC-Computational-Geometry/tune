import numpy as np
from mesh_model.mesh_analysis.quadmesh_analysis import isValidAction
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
        d111 = d11.get_beta(1)
        D = d111.get_node()

        # Template niveau 1
        template[n_darts - 1, 0] = nodes_scores[A.id]
        template[n_darts - 1, 1] = nodes_scores[B.id]
        template[n_darts - 1, 2] = nodes_scores[C.id]
        template[n_darts - 1, 3] = nodes_scores[D.id]


        E = [A,B,C,D]
        deep_captured = len(E)
        d2 = d.get_beta(2)
        d12 = d1.get_beta(2)
        d112 = d11.get_beta(2)
        d1112 = d111.get_beta(2)
        F = [d2, d12, d112, d1112]
        if deep>4:
            while len(E)<deep:
                df = F.pop(0)
                if df is not None:
                    df1 = df.get_beta(1)
                    df11 = df1.get_beta(1)
                    df111 = df11.get_beta(1)
                    F.append(df1)
                    F.append(df11)
                    F.append(df111)
                    N1, N2 = df11.get_node(), df111.get_node()
                    E.append(N1)
                    template[n_darts-1, len(E)-1] = nodes_scores[N1.id]
                    E.append(N2)
                    template[n_darts - 1, len(E)-1] = nodes_scores[N2.id]
                else:
                    E.extend([None,None])
                    #template[n_darts - 1, len(E) - 1] = -500 # dummy vertices are assigned to -500
                    #template[n_darts - 1, len(E) - 2] = -500 # dummy vertices are assigned to -500

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
        d111 = d11.get_beta(1)
        D = d111.get_node()

        # Template niveau 1
        template[n_darts - 1, 0] = nodes_scores[A.id]
        template[n_darts - 1, deep] = nodes_adjacency[A.id]
        template[n_darts - 1, 1] = nodes_scores[B.id]
        template[n_darts - 1, deep+1] = nodes_adjacency[B.id]
        template[n_darts - 1, 2] = nodes_scores[C.id]
        template[n_darts - 1, deep+2] = nodes_adjacency[C.id]
        template[n_darts - 1, 3] = nodes_scores[D.id]
        template[n_darts - 1, deep + 3] = nodes_adjacency[D.id]

        E = [A, B, C, D]
        deep_captured = len(E)
        d2 = d.get_beta(2)
        d12 = d1.get_beta(2)
        d112 = d11.get_beta(2)
        d1112 = d111.get_beta(2)
        F = [d2, d12, d112, d1112]
        if deep > 4:
            while len(E) < deep:
                df = F.pop(0)
                df1 = df.get_beta(1)
                df11 = df1.get_beta(1)
                df111 = df11.get_beta(1)
                F.append(df1)
                F.append(df11)
                F.append(df111)
                N1, N2 = df11.get_node(), df111.get_node()
                E.append(N1)
                template[n_darts - 1, len(E)] = nodes_scores[N1.id]
                template[n_darts - 1, deep + len(E)] = nodes_adjacency[N1.id]
                E.append(N2)
                template[n_darts - 1, len(E)] = nodes_scores[N2.id]
                template[n_darts - 1, deep + len(E)] = nodes_adjacency[N2.id]

    template = template[:n_darts, :]
    return template, dart_ids