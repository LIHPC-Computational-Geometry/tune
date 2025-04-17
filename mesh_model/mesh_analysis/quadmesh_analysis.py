import numpy as np

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.global_mesh_analysis import test_degree, on_boundary, adjacent_faces_id, degree

FLIP_CW = 0 # flip clockwise
FLIP_CCW = 1 # flip counterclockwise
SPLIT = 2
COLLAPSE = 3
CLEANUP = 4
TEST_ALL = 5 # test if all actions are valid
ONE_VALID = 6 # test if at least one action is valid


def isValidAction(mesh: Mesh, dart_id: int, action: int) -> (bool, bool):
    """
    Test if an action is valid. You can select the ype of action between {flip clockwise, flip counterclockwise, split, collapse, cleanup, all action, one action no matter wich one}.    :param mesh:
    :param mesh: a mesh
    :param dart_id: a dart on which to test the action
    :param action: an action type
    :return:
    """
    d = Dart(mesh, dart_id)
    if d.get_beta(2) is None:
        return False, True
    elif action == FLIP_CW:
        return isFlipCWOk(d)
    elif action == FLIP_CCW:
        return isFlipCCWOk(d)
    elif action == SPLIT:
        return isSplitOk(d)
    elif action == COLLAPSE:
        return isCollapseOk(d)
    elif action == CLEANUP:
        return isCleanupOk(d)
    elif action == TEST_ALL:
        topo, geo = isFlipCCWOk(d)
        if not (topo and geo):
            return False, False
        topo, geo = isFlipCWOk(d)
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
    elif action == ONE_VALID:
        topo_flip, geo_flip = isFlipCCWOk(d)
        if (topo_flip and geo_flip):
            return True, True
        topo_flip, geo_flip = isFlipCWOk(d)
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


def isFlipCCWOk(d: Dart) -> (bool, bool):
    mesh = d.mesh
    topo = True
    geo = True

    # if d is on boundary, flip is not possible
    if d.get_beta(2) is None:
        topo = False
        return topo, geo
    else:
        d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)

    # if degree will not too high
    if not test_degree(n5) or not test_degree(n3):
        topo = False
        return topo, geo

    # if two faces share two edges
    if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
        topo = False
        return topo, geo

    # check validity of the two modified quads
    geo = isValidQuad(n5, n6, n2, n3) and isValidQuad(n1, n5, n3, n4)

    return topo, geo

def isFlipCWOk(d: Dart) -> (bool, bool):
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
    if not test_degree(n4) or not test_degree(n6):
        topo = False
        return topo, geo

    if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
        topo = False
        return topo, geo
    geo = isValidQuad(n4, n6, n2, n3) and isValidQuad(n1, n5, n6, n4)

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
    geo = isValidQuad(n4, n1, n5, n10) and isValidQuad(n4, n10, n2, n3) and isValidQuad(n10, n5, n6, n2)
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

    if on_boundary(n1): # on_boundary(n3) or
        topo = False
        return topo, geo

    if (degree(n3)+degree(n1)-2) > 10:
        topo = False
        return topo, geo

    adjacent_faces_lst = []
    f1 = d2.get_face()
    adjacent_faces_lst.append(f1.id)
    d12 = d1.get_beta(2)
    if d12 is not None:
        f2 = d12.get_face()
        adjacent_faces_lst.append(f2.id)
    d112 = d11.get_beta(2)
    if d112 is not None:
        f3 = d112.get_face()
        adjacent_faces_lst.append(f3.id)
    d1112 = d111.get_beta(2)
    if d1112 is not None:
        f4 = d1112.get_face()
        adjacent_faces_lst.append(f4.id)

    # Check that there are no adjacent faces in common
    if len(adjacent_faces_lst) != len(set(adjacent_faces_lst)):
        topo = False
        return topo, geo

    adj_faces = adjacent_faces_id(n3)
    adj_faces.extend(adjacent_faces_id(n1))

    #If the opposite vertex is on the edge, it is not moved

    if on_boundary(n3):
        n10 = mesh.add_node( n3.x(), n3.y())
    else:
        n10 = mesh.add_node((n1.x() + n3.x()) / 2, (n1.y() + n3.y()) / 2)

    for f_id in adj_faces:
        if f_id != (d.get_face()).id:
            f = Face(mesh, f_id)
            df = f.get_dart()
            df1 = df.get_beta(1)
            df11 = df1.get_beta(1)
            df111 = df11.get_beta(1)
            A = df.get_node()
            B = df1.get_node()
            C = df11.get_node()
            D = df111.get_node()
            if A==n1 or A==n3:
                A=n10
            elif B==n1 or B==n3:
                B=n10
            elif C==n1 or C==n3:
                C=n10
            elif D==n1 or D==n3:
                D=n10

            if not isValidQuad(A, B, C, D):
                geo = False
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

    zero_count = sum(-1e-5<cp<1e-5 for cp in [cp_A, cp_B, cp_C, cp_D])
    if zero_count>=2:
        return False
    elif 0<= signe(cp_A)+signe(cp_B)+signe(cp_C)+signe(cp_D) <2 :
        return True
    else:
        return False
