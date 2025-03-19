from __future__ import annotations

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Node
from mesh_model.mesh_analysis.global_mesh_analysis import adjacent_darts, degree, mesh_check
from mesh_model.mesh_analysis.quadmesh_analysis import isFlipOk, isCollapseOk, isSplitOk, isCleanupOk


def flip_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return flip_edge(mesh, Node(mesh, id1), Node(mesh, id2))


def flip_edge(mesh: Mesh, n1: Node, n2: Node) -> (True, True, True):
    found, d = mesh.find_inner_edge(n1, n2)

    if found:
        topo, geo = isFlipOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
        return False, False, False

    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)

    f1 = d.get_face()
    f2 = d2.get_face()

    # Update beta 1
    d.set_beta(1, d11)
    d111.set_beta(1, d21)
    d21.set_beta(1, d)
    d2.set_beta(1, d211)
    d2111.set_beta(1, d1)
    d1.set_beta(1, d2)


    if n1.get_dart().id == d.id:
        n1.set_dart(d21)
    if n2.get_dart().id == d2.id:
        n2.set_dart(d1)

    if f1.get_dart().id == d1.id:
        f1.set_dart(d)

    if f2.get_dart().id == d21.id:
        f2.set_dart(d2)

    d.set_node(n5)
    d2.set_node(n3)
    d21.set_face(f1)
    d1.set_face(f2)

    return True, topo, geo


def split_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return split_edge(mesh, Node(mesh, id1), Node(mesh, id2))


def split_edge(mesh: Mesh, n1: Node, n2: Node) -> True:
    found, d = mesh.find_inner_edge(n1, n2)

    if found:
        topo, geo = isSplitOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
        return False, True, False

    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)
    d1112 = d111.get_beta(2)
    d212 = d21.get_beta(2)

    # create a new node in the middle of [n1, n2]
    N10 = mesh.add_node((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)

    # modify existing triangles
    d.set_node(N10)
    d21.set_node(N10)

    # create a new quadrangle
    f5 = mesh.add_quad(n1, n5, N10, n4)

    # update beta2 relations
    mesh.set_face_beta2(f5,[d111,d1112,d21,d212])

    adj_n1 = adjacent_darts(n1)
    adj_n2 = adjacent_darts(N10)

    deg1 = degree(n1)
    deg2 = degree(N10)

    return True, topo, geo


def collapse_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return collapse_edge(mesh, Node(mesh, id1), Node(mesh, id2))


def collapse_edge(mesh: Mesh, n1: Node, n2: Node) -> True:
    found, d = mesh.find_inner_edge(n1, n2)
    if found:
        topo, geo = isCollapseOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
        return False, False, False

    d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = mesh.active_quadrangles(d)

    d1112 = d111.get_beta(2)
    d12 = d1.get_beta(2)
    d112 = d11.get_beta(2)

    # Move n3 node in the middle of [n3, n1]
    n3.set_xy((n3.x() + n1.x()) / 2, (n1.y() + n3.y()) / 2)

    #Delete the face F5
    f5 = d.get_face()
    mesh.del_quad(d, d1, d11, d111, f5)

    n_from = n1
    n_to = n3
    adj_darts = adjacent_darts(n_from)

    for d in adj_darts:
        if d.get_node() == n_from:
            d.set_node(n_to)
    mesh.del_node(n_from)

    #Update beta2 relations
    if d2 is not None:
        d2.set_beta(2, d12)
        d21 = d2.get_beta(1)
        d21.set_node(n3)
    if d1112 is not None:
        d1112.set_beta(2, d112)
        d1112.set_node(n3)
    if d12 is not None:
        d12.set_beta(2, d2)
    if d112 is not None:
        d112.set_beta(2, d1112)

    return mesh_check(mesh), topo, geo


def cleanup_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return cleanup_edge(mesh, Node(mesh, id1), Node(mesh, id2))

def cleanup_edge(mesh: Mesh, n1: Node, n2: Node) -> True:
    found, d = mesh.find_inner_edge(n1, n2)
    if found:
        topo, geo = isCleanupOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
        return False, False, False

    parallel_darts = mesh.find_parallel_darts(d)

    last_dart = parallel_darts[-1]
    ld1 = last_dart.get_beta(1)
    ld11 = ld1.get_beta(1)
    ld111 = ld11.get_beta(1)
    last_node = ld111.get_node()
    node_to = ld11.get_node()
    adj_darts = adjacent_darts(last_node)
    mesh.del_node(last_node)

    for da in adj_darts:
        if da.get_node() == last_node:
            da.set_node(node_to)

    for d in parallel_darts:
        f = d.get_face()
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d111 = d11.get_beta(1)

        n_from = d.get_node()
        n_to = d1.get_node()

        # update beta 2 relations
        d12 = d1.get_beta(2)
        d1112 = d111.get_beta(2)
        if d1112 is not None:
            d1112.set_beta(2, d12)
        if d12 is not None:
            d12.set_beta(2, d1112)

        mesh.del_quad(d, d1, d11, d111, f)

        adj_darts = adjacent_darts(n_from)

        for d in adj_darts:
            if d.get_node() == n_from:
                d.set_node(n_to)
        mesh.del_node(n_from)

    return mesh_check(mesh), topo, geo


