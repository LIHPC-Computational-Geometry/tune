from __future__ import annotations

from mesh_model.mesh_analysis.trimesh_analysis import TriMeshTopoAnalysis, TriMeshGeoAnalysis
from mesh_model.mesh_struct.mesh_elements import Node, Dart
from view.mesh_plotter.mesh_plots import plot_mesh

"""
Actions triangulaires réalisées sur les maillages.
Chaque fonction retourne trois booléens:
* action_validity : Si l'action a été réalisée
* topo : Si l'action est valide topologiquement
* geo : si l'action est valide géométriquement
"""

def flip_edge_ids(mesh_analysis, id1: int, id2: int) -> True:
    return flip_edge(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2))


def flip_edge(mesh_analysis, n1: Node, n2: Node) -> True:
    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)

    if found:
        topo, geo = mesh_analysis.isSplitOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
        return False, False, True  # the geometrical criteria is True because if the dart is not found, it means it's a boundary dart


    d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh_analysis.mesh.active_triangles(d)

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

    check_mesh(mesh_analysis)
    return True, topo, geo


def split_edge_ids(mesh_analysis, id1: int, id2: int) -> True:
    return split_edge(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2))


def split_edge(mesh_analysis, n1: Node, n2: Node) -> True:
    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)

    if found:
        topo, geo = mesh_analysis.isSplitOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
            return False, False, True # the geometrical criteria is True because if the dart is not found, it means it's a boundary dart

    d2, d1, _, d21, _, n1, n2, n3, n4 = mesh_analysis.mesh.active_triangles(d)

    # create a new node in the middle of [n1, n2]
    N5 = mesh_analysis.mesh.add_node((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)

    # modify existing triangles
    d1.set_node(N5)
    d21.set_node(N5)
    d.set_beta(1, d1)
    d2.set_beta(1, d21)

    # create 2 new triangles
    F3 = mesh_analysis.mesh.add_triangle(n2, n3, N5)
    F4 = mesh_analysis.mesh.add_triangle(N5, n1, n4)

    # update beta2 relations
    mesh_analysis.mesh.set_face_beta2(F3, [d1, d2, d1.get_beta(2)])
    mesh_analysis.mesh.set_face_beta2(F4, [d, d21, d21.get_beta(2)])

    check_mesh(mesh_analysis)
    return True, topo, geo


def collapse_edge_ids(mesh_analysis, id1: int, id2: int) -> True:
    return collapse_edge(mesh_analysis, Node(mesh_analysis.mesh, id1), Node(mesh_analysis.mesh, id2))


def collapse_edge(mesh_analysis, n1: Node, n2: Node) -> True:
    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)

    if found:
        topo, geo = mesh_analysis.isCollapseOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
        return False, False, True  # the geometrical criteria is True because if the dart is not found, it means it's a boundary dart

    _, d1, d11, d21, d211, n1, n2, n3, n4 = mesh_analysis.mesh.active_triangles(d)

    d212 = d21.get_beta(2) #T1
    d2112 = d211.get_beta(2) #T2
    d12 = d1.get_beta(2) #T3
    d112 = d11.get_beta(2) #T4

    #Delete the darts around selected dart
    mesh_analysis.mesh.del_adj_triangles(d)

    #Move n1 node in the middle of [n1, n2]
    n1.set_xy((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)
    i = 0

    #Check if nodes n3 and n4 are not linked to deleted dart

    if n3.get_dart().id == d11.id:
        if d12 is not None:
            n3.set_dart(d12)
        else:
            n3.set_dart(d112.get_beta(1))
    if n4.get_dart().id == d211.id:
        if d212 is not None:
            n4.set_dart(d212)
        else:
            n4.set_dart(d2112.get_beta(1))

    #Update node relations
    if d12 is not None:
        d121 = d12.get_beta(1)
        d121.set_node(n1)
        ds = d121
        while ds is not None and ds != d2112:
            i+=1
            d2s = ds.get_beta(2)
            if d2s is None:
                ds = d2112
                while ds is not None:
                    i+=1
                    ds.set_node(n1)
                    ds1 = ds.get_beta(1)
                    ds11 = ds1.get_beta(1)
                    ds = ds11.get_beta(2)
                    if i > 30:
                        i = 0
                        plot_mesh(mesh_analysis.mesh)
            else:
                ds = d2s.get_beta(1)
                ds.set_node(n1)
            if i>30:
                i=0
                plot_mesh(mesh_analysis.mesh)
    """
    elif d12 is None and d2112 is not None:
        d2112.set_node(n1)
        ds = (d2112.get_beta(1)).get_beta(1)
        ds2 = ds.get_beta(2)
        while ds2 is not None:
            ds2.set_node(n1)
            ds = (ds2.get_beta(1)).get_beta(1)
            ds2 = ds.get_beta(2)
    """
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
    mesh_analysis.mesh.del_node(n2)

    check_mesh(mesh_analysis)
    return True, topo, geo

def check_mesh(mesh_analysis):
    for dart_info in mesh_analysis.mesh.active_darts():
        #Check beta2 relation
        d = dart_info[0]
        d2 = dart_info[2]
        if d2 >= 0 and mesh_analysis.mesh.dart_info[d2, 0] < 0:
            raise ValueError("error beta2")
        elif d2 >= 0 and mesh_analysis.mesh.dart_info[d2, 2] != d:
            raise ValueError("error beta2")

        #Check beta1
        d = Dart(mesh_analysis.mesh, d)
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        if d11.get_beta(1)!=d :
            raise ValueError("error beta1")
