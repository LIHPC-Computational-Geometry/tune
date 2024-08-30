from __future__ import annotations

from model.mesh_struct.mesh import Mesh
from model.mesh_struct.mesh_elements import Dart, Node
from model.mesh_analysis import degree, isFlipOk, isCollapseOk

import numpy as np


def flip_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return flip_edge(mesh, Node(mesh, id1), Node(mesh, id2))


def flip_edge(mesh: Mesh, n1: Node, n2: Node) -> True:
    found, d = mesh.find_inner_edge(n1, n2)

    if not found or not isFlipOk(d):
        return False

    d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)

    test_degree(n3)
    test_degree(n4)

    f1 = d.get_face()
    f2 = d2.get_face()

    d.set_beta(1, d211)
    d2.set_beta(1, d11)
    d21.set_beta(1, d2)
    d1.set_beta(1, d)
    d11.set_beta(1, d21)
    d211.set_beta(1, d1)

    if n1.get_dart().id == d.id:
        n1.set_dart(d21)
    if n2.get_dart().id == d2.id:
        n2.set_dart(d1)

    if f1.get_dart().id == d11.id:
        f1.set_dart(d)

    if f2.get_dart().id == d211.id:
        f2.set_dart(d2)

    d.set_node(n3)
    d2.set_node(n4)
    d211.set_face(f1)
    d11.set_face(f2)
    return True


def split_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return split_edge(mesh, Node(mesh, id1), Node(mesh, id2))


def split_edge(mesh: Mesh, n1: Node, n2: Node) -> True:
    found, d = mesh.find_inner_edge(n1, n2)
    if not found:
        return False

    d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)
    test_degree(n3)
    test_degree(n4)

    # create a new node in the middle of [n1, n2]
    N5 = mesh.add_node((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)

    # modify existing triangles
    d1.set_node(N5)
    d21.set_node(N5)
    d.set_beta(1, d1)
    d2.set_beta(1, d21)

    # create 2 new triangles
    F3 = mesh.add_triangle(n2, n3, N5)
    F4 = mesh.add_triangle(N5, n1, n4)

    # update beta2 relations
    mesh.set_face_beta2(F3, (d1, d2))
    d2b2 = d2.get_beta(2)
    d2b21 = d2b2.get_beta(1)
    mesh.set_beta2(d2b21)
    mesh.set_face_beta2(F4, (d, d21))
    db2 = d.get_beta(2)
    db21 = db2.get_beta(1)
    mesh.set_beta2(db21)
    return True


def collapse_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return collapse_edge(mesh, Node(mesh, id1), Node(mesh, id2))


def collapse_edge(mesh: Mesh, n1: Node, n2: Node) -> True:
    found, d = mesh.find_inner_edge(n1, n2)
    if not found or not isCollapseOk(d):
        return False

    d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)

    #T1
    d212 = d21.get_beta(2)

    #T2
    d2112 = d211.get_beta(2)

    #T3
    d12 = d1.get_beta(2)

    #T4
    d112 = d11.get_beta(2)

    #Delete the darts around selected dart
    delete_triangles(mesh, d)

    #move n1 node in the middle of [n1, n2]
    n1.set_xy((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)

    #update node relations
    if d12 is not None:
        d121 = d12.get_beta(1)
        d121.set_node(n1)
        ds = d121
        while ds is not None and ds != d2112:
            d2s = ds.get_beta(2)
            if d2s is None:
                ds = d2112
                while ds is not None:
                    ds.set_node(n1)
                    ds1 = ds.get_beta(1)
                    ds11 = ds1.get_beta(1)
                    ds = ds11.get_beta(2)
            else:
                ds = d2s.get_beta(1)
                ds.set_node(n1)

    #update beta2 relations
    if d112 is not None:
        d112.set_beta(2, d12)
    if d12 is not None:
        d12.set_beta(2, d112)

    if d212 is not None:
        d212.set_beta(2, d2112)
    if d2112 is not None:
        d2112.set_beta(2, d212)

    #delete n2 node
    mesh.del_node(n2)

    return True

def delete_triangles(mesh: Mesh, d: Dart) -> None:
    d2 = d.get_beta(2)
    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    d21 = d2.get_beta(1)
    d211 = d21.get_beta(1)

    f1 = d.get_face()
    f2 = d2.get_face()

    mesh.del_triangle(d, d1, d11, f1)
    mesh.del_triangle(d2, d21, d211, f2)



def test_degree(n: Node) -> bool:
    """
    Verify that the degree of a vertex is lower than 10
    :param n: a Node
    :return: True if the degree is lower than 10, False otherwise
    """
    if degree(n) > 10:
        return False
    else:
        return True