import numpy as np
from mesh_model.mesh_struct.mesh_elements import Dart, Node
from mesh_model.mesh_struct.mesh import Mesh


def get_x(m_analysis, n_darts_selected: int, deep :int, quality: bool, restricted:bool, nodes_scores: list[int], nodes_adjacency: list[int]):
    mesh = m_analysis.mesh
    if quality:
        template, darts_id = get_template_with_quality(m_analysis, deep)
    else:
        template, darts_id = get_template(m_analysis, deep, nodes_scores)

    if restricted:
        darts_to_delete = []
        darts_id = []
        for i, d_info in enumerate(mesh.active_darts()):
            d_id = d_info[0]
            if d_info[2] == -1 or not m_analysis.isValidAction(d_info[0], 4)[0]:  # test the validity of all action type
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


def get_template(m_analysis, deep: int, nodes_scores):
    size = len(m_analysis.mesh.dart_info)
    template = np.zeros((size, deep), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in m_analysis.mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(m_analysis.mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()

        # Template niveau 1
        template[n_darts - 1, 0] = C.get_score()
        template[n_darts - 1, 1] = A.get_score()
        template[n_darts - 1, 2] = B.get_score()

        if deep>3:
            # template niveau 2 deep = 6
            n_id = m_analysis.find_template_opposite_node(d)
            if n_id is not None:
                template[n_darts - 1, 3] = n_id.get_score()
            n_id = m_analysis.find_template_opposite_node(d1)
            if n_id is not None:
                template[n_darts - 1, 4] = n_id.get_score()
            n_id = m_analysis.find_template_opposite_node(d11)
            if n_id is not None:
                template[n_darts - 1, 5] = n_id.get_score()

        if deep>6:
            # template niveau 3 - deep = 12
            d2, d1, d11, d21, d211, n1, n2, n3, n4 = m_analysis.mesh.active_triangles(d)
            #Triangle F2
            n_id = m_analysis.find_template_opposite_node(d21)
            if n_id is not None:
                template[n_darts - 1, 6] = n_id.get_score()
            n_id = m_analysis.find_template_opposite_node(d211)
            if n_id is not None:
                template[n_darts - 1, 7] = n_id.get_score()
            # Triangle T3
            d12 = d1.get_beta(2)
            d121 = d12.get_beta(1)
            d1211 = d121.get_beta(1)
            n_id = m_analysis.find_template_opposite_node(d121)
            if n_id is not None:
                template[n_darts - 1, 8] = n_id.get_score()
            n_id = m_analysis.find_template_opposite_node(d1211)
            if n_id is not None:
                template[n_darts - 1, 9] = n_id.get_score()
            # Triangle T4
            d112 = d11.get_beta(2)
            d1121 = d112.get_beta(1)
            d11211 = d1121.get_beta(1)
            n_id = m_analysis.find_template_opposite_node(d1121)
            if n_id is not None:
                template[n_darts - 1, 10] = n_id.get_score()
            n_id = m_analysis.find_template_opposite_node(d11211)
            if n_id is not None:
                template[n_darts - 1, 11] = n_id.get_score()

    template = template[:n_darts, :]

    return template, dart_ids

def get_template_with_quality(m_analysis, deep: int):
    """
    Create a template matrix representing the entire mesh, composed of:

    * Node scores: i.e. the difference between ideal adjacency and actual adjacency.
    * Dart surrounding quality: a measure of the geometric quality around each dart.

    Each column in the matrix corresponds to the local surrounding of a dart,
    including the scores of its surrounding nodes and its associated quality.

    :param m_analysis: mesh to analyze
    :param deep: observation deep (how many nodes observed on each dart surrounding)
    :return: template matrix
    """
    size = len(m_analysis.mesh.dart_info)
    template = np.zeros((size, deep*2), dtype=np.int64)
    dart_ids = []
    n_darts = 0

    for d_info in m_analysis.mesh.active_darts():
        n_darts += 1
        d_id = d_info[0]
        dart_ids.append(d_id)
        d = Dart(m_analysis.mesh, d_id)
        A = d.get_node()
        d1 = d.get_beta(1)
        B = d1.get_node()
        d11 = d1.get_beta(1)
        C = d11.get_node()

        # Template niveau 1
        template[n_darts - 1, 0] = C.get_score()
        template[n_darts - 1, deep] = d.get_quality()
        template[n_darts - 1, 1] = A.get_score()
        template[n_darts - 1, deep+1] = d1.get_quality()
        template[n_darts - 1, 2] = B.get_score()
        template[n_darts - 1, deep+2] = d11.get_quality()

        if deep>3:
            # template niveau 2
            n_id = m_analysis.find_template_opposite_node(d)
            if n_id is not None:
                n = Node(m_analysis.mesh, n_id)
                template[n_darts - 1, 3] = n.get_score()
                template[n_darts - 1, deep+3] = d.get_quality() #quality around dart d is equivalent to quality around dart d2
            n_id = m_analysis.find_template_opposite_node(d1)
            if n_id is not None:
                n = Node(m_analysis.mesh, n_id)
                template[n_darts - 1, 3] = n.get_score()
                template[n_darts - 1, deep+4] = d1.get_quality()
            n_id = m_analysis.find_template_opposite_node(d11)
            if n_id is not None:
                n = Node(m_analysis.mesh, n_id)
                template[n_darts - 1, 3] = n.get_score()
                template[n_darts - 1, deep+5] = d11.get_quality()

        if deep>6:
            # template niveau 3 - deep = 12
            if d.get_beta(2) is not None:
                d2, d1, d11, d21, d211, n1, n2, n3, n4 = m_analysis.mesh.active_triangles(d)
                #Triangle F2
                n_id = m_analysis.find_template_opposite_node(d21)
                if n_id is not None:
                    n = Node(m_analysis.mesh, n_id)
                    template[n_darts - 1, 3] = n.get_score()
                    template[n_darts - 1, deep+6] = d21.get_quality()
                n_id = m_analysis.find_template_opposite_node(d211)
                if n_id is not None:
                    n = Node(m_analysis.mesh, n_id)
                    template[n_darts - 1, 3] = n.get_score()
                    template[n_darts - 1, deep+7] = d211.get_quality()
            # Triangle T3
            d12 = d1.get_beta(2)
            if d12 is not None:
                d121 = d12.get_beta(1)
                d1211 = d121.get_beta(1)
                n_id = m_analysis.find_template_opposite_node(d121)
                if n_id is not None:
                    n = Node(m_analysis.mesh, n_id)
                    template[n_darts - 1, 3] = n.get_score()
                    template[n_darts - 1, deep+8] = d121.get_quality()
                n_id = m_analysis.find_template_opposite_node(d1211)
                if n_id is not None:
                    n = Node(m_analysis.mesh, n_id)
                    template[n_darts - 1, 3] = n.get_score()
                    template[n_darts - 1, deep+9] = d1211.get_quality()
            # Triangle T4
            d112 = d11.get_beta(2)
            if d112 is not None:
                d1121 = d112.get_beta(1)
                d11211 = d1121.get_beta(1)
                n_id = m_analysis.find_template_opposite_node(d1121)
                if n_id is not None:
                    n = Node(m_analysis.mesh, n_id)
                    template[n_darts - 1, 3] = n.get_score()
                    template[n_darts - 1, deep+10] = d1121.get_quality()
                n_id = m_analysis.find_template_opposite_node(d11211)
                if n_id is not None:
                    n = Node(m_analysis.mesh, n_id)
                    template[n_darts - 1, 3] = n.get_score()
                    template[n_darts - 1, deep+11] = d11211.get_quality()

    template = template[:n_darts, :]
    return template, dart_ids