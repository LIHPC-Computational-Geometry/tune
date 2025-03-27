from math import sqrt, degrees, radians, cos, sin, acos
import numpy as np

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.global_mesh_analysis import test_degree, on_boundary, get_angle_by_coord, degree, \
    get_boundary_angle, adjacent_darts, adjacent_faces


def global_score(m: Mesh):
    """
    Calculate the overall mesh score. The mesh cannot achieve a better score than the ideal one.
    And the current score is the mesh score.
    :param m: the mesh to be analyzed
    :return: 4 return: a list of the nodes score, the current mesh score and the ideal mesh score, and the adjacency
    """
    mesh_ideal_score = 0
    mesh_score = 0
    nodes_score = []
    nodes_adjacency = []
    for i in range(len(m.nodes)):
        if m.nodes[i, 2] >= 0:
            n_id = i
            node = Node(m, n_id)
            n_score, adjacency= score_calculation(node)
            nodes_score.append(n_score)
            nodes_adjacency.append(adjacency)
            mesh_ideal_score += n_score
            mesh_score += abs(n_score)
        else:
            nodes_score.append(0)
            nodes_adjacency.append(6)
    return nodes_score, mesh_score, mesh_ideal_score, nodes_adjacency


def score_calculation(n: Node) -> (int, int):
    """
    Function to calculate the irregularity of a node in the mesh.
    :param n: a node in the mesh.
    :return: the irregularity of the node
    """
    adjacency = degree(n)
    if on_boundary(n):
        angle = get_boundary_angle(n)
        ideal_adjacency = max(round(angle/90)+1, 2)
    else:
        ideal_adjacency = 360/90

    return ideal_adjacency-adjacency, adjacency


def isValidAction(mesh: Mesh, dart_id: int, action: int) -> (bool, bool):
    flip = 0
    split = 1
    collapse = 2
    cleanup =3
    test_all = 4
    one_valid = 5
    d = Dart(mesh, dart_id)
    if d.get_beta(2) is None:
        return False, True
    elif action == flip:
        return isFlipOk(d)
    elif action == split:
        return isSplitOk(d)
    elif action == collapse:
        return isCollapseOk(d)
    elif action == cleanup:
        return isCleanupOk(d)
    elif action == test_all:
        topo, geo = isFlipOk(d)
        if not (topo and geo):
            return False, False
        topo, geo = isSplitOk(d)
        if not (topo and geo):
            return False, False
        topo, geo = isCollapseOk(d)
        if not (topo and geo):
            return False, False
        elif topo and geo:
            return True, True
    elif action == one_valid:
        topo_flip, geo_flip = isFlipOk(d)
        if (topo_flip and geo_flip):
            return True, True
        topo_split, geo_split = isSplitOk(d)
        if (topo_split and geo_split):
            return True, True
        topo_collapse, geo_collapse = isCollapseOk(d)
        if (topo_collapse and geo_collapse):
            return True, True
        return False, False
    else:
        raise ValueError("No valid action")


def isFlipOk(d: Dart) -> (bool, bool):
    mesh = d.mesh
    topo = True
    geo = True
    # if d is on boundary, flip is not possible
    if d.get_beta(2) is None:
        topo = False
        return topo, geo
    else:
        d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)
    # if degree are
    if not test_degree(n5) or not test_degree(n3):
        topo = False
        return topo, geo

    if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
        topo = False
        return topo, geo
    topo = isValidQuad(n5, n6, n2, n3) and isValidQuad(n1, n5, n3, n4)

    """
    # Check angle at d limits to avoid edge reversal
    angle_A = get_angle_by_coord(n5.x(), n5.y(), n1.x(), n1.y(), n3.x(), n3.y())

    if angle_A <= 90 or angle_A >= 180:
        topo = False
        return topo, geo
    """

    return topo, geo


def isSplitOk(d: Dart) -> (bool, bool):
    mesh = d.mesh
    topo = True
    geo = True
    if d.get_beta(2) is None:
        topo = False
        return topo, geo
    else:
        d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)

    if not test_degree(n4) or not test_degree(n2):
        topo = False
        return topo, geo

    if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
        topo = False
        return topo, geo

    n10 = mesh.add_node((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)
    topo = isValidQuad(n4, n1, n5, n10) and isValidQuad(n4, n10, n2, n3) and isValidQuad(n10, n5, n6, n2)
    mesh.del_node(n10)
    return topo, geo


def isCollapseOk(d: Dart) -> (bool, bool):
    mesh = d.mesh
    topo = True
    geo = True
    if d.get_beta(2) is None:
        topo = False
        return topo, geo
    else:
        d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)

    if on_boundary(n3) or on_boundary(n1):
        topo = False
        return topo, geo

    if not test_degree(n3):
        topo = False
        return topo, geo

    f1 = d2.get_face()
    f2 = (d1.get_beta(2)).get_face()
    f3 = (d11.get_beta(2)).get_face()
    f4 = (d111.get_beta(2)).get_face()
    adjacent_faces_lst=[f1.id, f2.id, f3.id, f4.id]

    # Check that there are no adjacent faces in common
    if len(adjacent_faces_lst) != len(set(adjacent_faces_lst)):
        topo = False
        return topo, geo

    adj_faces = adjacent_faces(n3)
    adj_faces.extend(adjacent_faces(n1))
    n10 = mesh.add_node((n1.x() + n3.x()) / 2, (n1.y() + n3.y()) / 2)

    for f in adj_faces:
        d = f.get_dart()
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d111 = d11.get_beta(1)
        A = d.get_node()
        B = d1.get_node()
        C = d11.get_node()
        D = d111.get_node()
        if A==n1 or A==n3:
            A=n10
        elif B==n1 or B==n3:
            B=n10
        elif C==n1 or C==n3:
            C=n10
        elif D==n1 or D==n3:
            D=n10

        if not isValidQuad(A, B, C, D):
            topo = False
            mesh.del_node(n10)
            return topo, geo

    mesh.del_node(n10)
    return topo, geo


def isCleanupOk(d: Dart) -> (bool, bool):
    topo = True
    geo = True
    if d.get_beta(2) is None:
        topo = False
    mesh = d.mesh
    parallel_darts = mesh.find_parallel_darts(d)
    for d in parallel_darts:
        d111 = ((d.get_beta(1)).get_beta(1)).get_beta(1)
        if d111.get_beta(2) is None:
            topo = False
            return topo, geo
    return topo, geo


def isTruncated(m: Mesh, darts_list)-> bool:
    for d_id in darts_list:
        if isValidAction(m, d_id, 4)[0]:
            return False
    return True

def cross_product(vect_AB, vect_AC):
    """ Return the cross product between AB et AC.
        0 means A, B and C are coolinear
        > 0 mean A, B and C are "sens des aiguilles d'une montre"
        < 0 sens inverse
    """
    val = vect_AB[0] * vect_AC[1] - vect_AB[1] * vect_AC[0]
    return val

def signe(a: int):
    if a<=0:
        return 0
    else:
        return 1

def isValidQuad(A: Node, B: Node, C: Node, D: Node):
    u1 = np.array([B.x() - A.x(), B.y() - A.y()]) # vect(AB)
    u2 = np.array([C.x() - B.x(), C.y() - B.y()]) # vect(BC)
    u3 = np.array([D.x() - C.x(), D.y() - C.y()]) # vect(CD)
    u4 = np.array([A.x() - D.x(), A.y() - D.y()]) # vect(DA)

    # Checking for near-zero vectors (close to (0,0))
    if (np.linalg.norm(u1) < 1e-5 or
            np.linalg.norm(u2) < 1e-5 or
            np.linalg.norm(u3) < 1e-5 or
            np.linalg.norm(u4) < 1e-5):
        return False  # Quad invalid because one side is almost zero

    cp_A = cross_product(-1*u4, u1)
    cp_B = cross_product(-1*u1, u2)
    cp_C = cross_product(-1*u2, u3)
    cp_D = cross_product(-1*u3, u4)

    """
    if cp_A<=0 and cp_B<=0 and cp_C<=0 and cp_D<=0:
        return True
    else:
        return False

    """
    if 0< signe(cp_A)+signe(cp_B)+signe(cp_C)+signe(cp_D) <2 :
        return True
    else:
        return False


def orientation(p, q, r):
    """ Calcule l'orientation de trois points.
        Retourne :
        0 si colinéaire,
        1 si sens anti-horaire,
        2 si sens horaire.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def do_intersect(p1, q1, p2, q2):
    """ Vérifie si les segments [p1,q1] et [p2,q2] s'intersectent """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Cas général : les orientations sont différentes
    if o1 != o2 and o3 != o4:
        return True

    return False  # Pas d'intersection

def is_self_intersecting_quadrilateral(A, B, C, D):
    """ Vérifie si le quadrilatère ABCD est croisé """
    return do_intersect(A, C, B, D)  # Vérifie l'intersection des diagonales


"""
d12 = d1.get_beta(2)
d112 = d11.get_beta(2)
d1112 = d111.get_beta(2)
n7 = ((d1112.get_beta(1)).get_beta(1)).get_node()
n8 = ((d112.get_beta(1)).get_beta(1)).get_node()
n9 = (((d112.get_beta(1)).get_beta(1)).get_beta(1)).get_node()
n11 = ((d12.get_beta(1)).get_beta(1)).get_node()
n10 = mesh.add_node((n1.x() + n3.x()) / 2, (n1.y() + n3.y()) / 2)
topo = isValidQuad(n10, n5, n6, n2) and isValidQuad(n4, n10, n2, n3) and isValidQuad(n10, n5, n6, n2)
"""