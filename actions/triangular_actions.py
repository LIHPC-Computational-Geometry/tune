from __future__ import annotations

from model.mesh_struct.mesh import Mesh
from model.mesh_struct.mesh_elements import Dart, Node


def flip_edge_ids(mesh: Mesh, id1: int, id2: int) -> True:
    return flip_edge(mesh, Node(mesh, id1), Node(mesh, id2))


def flip_edge(mesh: Mesh, n1: Node, n2: Node) -> True:
    found, d = mesh.find_inner_edge(n1, n2)
    if not found:
        return False
    d2 = d.get_beta(2)
    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    d21 = d2.get_beta(1)
    d211 = d21.get_beta(1)
    n1 = d.get_node()
    n2 = d2.get_node()
    n3 = d11.get_node()
    n4 = d211.get_node()


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
    d2 = d.get_beta(2)
    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    d21 = d2.get_beta(1)
    d211 = d21.get_beta(1)
    N1 = d.get_node()
    N2 = d2.get_node()
    N3 = d11.get_node()
    N4 = d211.get_node()

    # create a new node in the middle of [n1, n2]
    N5 = mesh.add_node((N1.x() + N2.x()) / 2, (N1.y() + N2.y()) / 2)

    # modify existing triangles
    d1.set_node(N5)
    d21.set_node(N5)
    d.set_beta(1, d1)
    d2.set_beta(1, d21)

    # create 2 new triangles
    F3 = mesh.add_triangle(N2, N3, N5)
    F4 = mesh.add_triangle(N5, N1, N4)

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