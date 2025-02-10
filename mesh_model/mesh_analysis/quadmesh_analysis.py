from math import sqrt, degrees, radians, cos, sin, acos
import numpy as np

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.mesh_analysis import test_degree, on_boundary, get_angle_by_coord


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

    if not test_degree(n5) or not test_degree(n3):
        topo = False
        return topo, geo

    # Check angle at d limits to avoid edge reversal
    angle_A = get_angle_by_coord(n5.x(), n5.y(), n1.x(), n1.y(), n3.x(), n3.y())

    if angle_A <= 90 or angle_A >= 180:
        topo = False
        return topo, geo

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