from __future__ import annotations

from copy import deepcopy

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
    """
    Lors de la bascule d'arete, les relations beta2 ne sont pas impactées, seulement beta1.
    :param mesh_analysis:
    :param n1:
    :param n2:
    :return:
    """
    valid_action = True
    found, d = mesh_analysis.mesh.find_inner_edge(n1, n2)
    mesh_before = deepcopy(mesh_analysis.mesh)
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

    topo = check_mesh(mesh_analysis, mesh_before)
    if not topo:
        mesh_analysis.mesh = deepcopy(mesh_before)
        valid_action = False
    return valid_action, topo, geo


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
    mesh = mesh_analysis.mesh
    found, d = mesh.find_inner_edge(n1, n2)

    if found:
        topo, geo = mesh_analysis.isCollapseOk(d)
        if not geo or not topo:
            return False, topo, geo
    else:
        return False, False, True  # the geometrical criteria is True because if the dart is not found, it means it's a boundary dart

    d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)

    d212 = d21.get_beta(2) #T1
    d2112 = d211.get_beta(2) #T2
    d12 = d1.get_beta(2) #T3
    if not mesh.is_dart_active(d12):
        print("error")
    d112 = d11.get_beta(2) #T4

    #Delete the darts around selected dart
    mesh_analysis.mesh.del_adj_triangles(d)

    #Move n1 node in the middle of [n1, n2]
    n1.set_xy((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)
    i = 0

    #Check if nodes n3 and n4 are not linked to deleted dart

    if n3.get_dart().id == d11.id:
        if mesh.is_dart_active(d12):
            n3.set_dart(d12)
        else:
            n3.set_dart(d112.get_beta(1))
    if n4.get_dart().id == d211.id:
        if mesh.is_dart_active(d212):
            n4.set_dart(d212)
        else:
            n4.set_dart(d2112.get_beta(1))
    if n1.get_dart().id == d.id or n1.get_dart().id == d21.id:
        if mesh.is_dart_active(d112):
            n1.set_dart(d112)
        else:
            n1.set_dart(d2112)


    #Update node relations
    if mesh.is_dart_active(d12):
        d121 = d12.get_beta(1)
        d121.set_node(n1)
        ds = d121
        while ds is not None and ds != d2112:
            i+=1
            d2s = ds.get_beta(2)
            if not mesh.is_dart_active(d2s):
                ds = d2112
                while mesh.is_dart_active(ds):
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
    if mesh.is_dart_active(d112):
        d112.set_beta(2, d12)
    if mesh.is_dart_active(d12):
        d12.set_beta(2, d112)

    if mesh.is_dart_active(d212):
        d212.set_beta(2, d2112)
    if mesh.is_dart_active(d2112):
        d2112.set_beta(2, d212)

    #delete n2 node
    mesh_analysis.mesh.del_node(n2)

    check_mesh(mesh_analysis)
    return True, topo, geo

def check_mesh(mesh_analysis, mesh_before=None) -> bool:
    for dart_info in mesh_analysis.mesh.active_darts():
        #Check beta2 relation
        d = dart_info[0]
        d2 = dart_info[2]
        # if associated twin dart no longer exist
        if d2 >= 0 and mesh_analysis.mesh.dart_info[d2, 0] < 0:
            return False
        # if beta2 relation is not symetrical
        elif d2 >= 0 and mesh_analysis.mesh.dart_info[d2, 2] != d:
            return False
        # null dart
        elif d2>=0 and mesh_analysis.mesh.dart_info[d2, 3] == mesh_analysis.mesh.dart_info[d, 3]:
            return False
        #if adjacent face is the same
        elif  d2>=0 and mesh_analysis.mesh.dart_info[d2, 4] == mesh_analysis.mesh.dart_info[d, 4]:
            return False


        d1 = mesh_analysis.mesh.dart_info[d,1]
        d11 = mesh_analysis.mesh.dart_info[d1,1]

        #Check beta1
        if  mesh_analysis.mesh.dart_info[d11,1]!=d :
            return False

        if d2 >= 0 :
            d = Dart(mesh_analysis.mesh, d)
            d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh_analysis.mesh.active_triangles(d)
            if len(set([n1.id, n2.id, n3.id, n4.id])) < 4:
                return False
        return True


def check_mesh_debug(mesh_analysis, mesh_before=None)->True:
    for dart_info in mesh_analysis.mesh.active_darts():
        #Check beta2 relation
        d = dart_info[0]
        d2 = dart_info[2]
        # if associated twin dart no longer exist
        if d2 >= 0 and mesh_analysis.mesh.dart_info[d2, 0] < 0:
            plot_mesh(mesh_analysis.mesh)
            if mesh_before is not None:
                plot_mesh(mesh_before)
            raise ValueError("error beta2")
        # if beta2 relation is not symetrical
        elif d2 >= 0 and mesh_analysis.mesh.dart_info[d2, 2] != d:
            plot_mesh(mesh_analysis.mesh)
            if mesh_before is not None:
                plot_mesh(mesh_before)
            raise ValueError("error beta2")
        # null dart
        elif d2>=0 and mesh_analysis.mesh.dart_info[d2, 3] == mesh_analysis.mesh.dart_info[d, 3]:
            plot_mesh(mesh_analysis.mesh)
            if mesh_before is not None:
                plot_mesh(mesh_before)
            raise ValueError("same node for twin darts")
        #if adjacent face is the same
        elif  d2>=0 and mesh_analysis.mesh.dart_info[d2, 4] == mesh_analysis.mesh.dart_info[d, 4]:
            plot_mesh(mesh_analysis.mesh)
            if mesh_before is not None:
                plot_mesh(mesh_before)
            raise ValueError("same adjacent face")


        d1 = mesh_analysis.mesh.dart_info[d,1]
        d11 = mesh_analysis.mesh.dart_info[d1,1]

        #Check beta1
        if  mesh_analysis.mesh.dart_info[d11,1]!=d :
            plot_mesh(mesh_analysis.mesh)
            if mesh_before is not None:
                plot_mesh(mesh_before)
            raise ValueError("error beta1")

        if d2 >= 0 :
            d = Dart(mesh_analysis.mesh, d)
            d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh_analysis.mesh.active_triangles(d)

            if len(set([n1.id, n2.id, n3.id, n4.id])) < 4:
                plot_mesh(mesh_analysis.mesh)
                if mesh_before is not None:
                    plot_mesh(mesh_before)
                raise ValueError("same traingle for two faces")