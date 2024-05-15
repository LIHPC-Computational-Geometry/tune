from math import sqrt, degrees
import numpy as np

from model.mesh_struct.mesh_elements import Dart, Node
from model.mesh_struct.mesh import Mesh


def global_score(m: Mesh) -> (int, int):
    """
    Calculate the overall mesh score. The mesh cannot achieve a better score than the ideal one.
    And the current score is the mesh score.
    :param m: the mesh to be analyzed
    :return: three return values: a list of the nodes score, the current mesh score and the ideal mesh score
    """
    mesh_ideal_score = 0
    mesh_score = 0
    nodes_score = []
    for i in range(len(m.nodes)):
        n_id = i
        node = Node(m, n_id)
        n_score = score_calculation(node)
        nodes_score.append(n_score)
        mesh_ideal_score += n_score
        mesh_score += abs(n_score)
    return nodes_score, mesh_score, mesh_ideal_score


def score_calculation(n: Node) -> int:
    """
    Function to calculate the irregularity of a node in the mesh.
    :param n: a node in the mesh.
    :return: the irregularity of the node
    """
    adjacency = degree(n)
    if on_boundary(n):
        angle = get_boundary_angle(n)
        ideal_adjacency = max(round(angle/60)+1, 2)
    else:
        ideal_adjacency = 360/60

    return ideal_adjacency-adjacency


def get_angle(d1: Dart, d2: Dart, n: Node) -> float:
    """
    Function to calculate the angle of the boundary at the node n.
    The angle is named ABC and node n is at point A.
    :param d1: the first boundary dart.
    :param d2: the second boundary dart.
    :param n: the boundary node
    :return:
    """
    if d1.get_node() == n:
        A = n
        B = d1.get_beta(1).get_node()
        C = d2.get_node()

    else:
        A = n
        B = d2.get_beta(1).get_node()
        C = d1.get_node()
        if d2.get_node() != A:
            raise ValueError("Angle error")

    vect_AB = (B.x() - A.x(), B.y() - A.y())
    vect_AC = (C.x() - A.x(), C.y() - A.y())
    dist_AB = sqrt(vect_AB[0]**2 + vect_AB[1]**2)
    dist_AC = sqrt(vect_AC[0]**2 + vect_AC[1]**2)
    angle = np.arccos(np.dot(vect_AB, vect_AC)/(dist_AB*dist_AC))
    return degrees(angle)


def get_boundary_angle(n: Node) -> float:
    adj_darts_list = adjacent_darts(n)
    boundary_darts = []
    for d in adj_darts_list:
        d_twin = d.get_beta(2)
        if d_twin is None:
            boundary_darts.append(d)
    if len(boundary_darts) > 3:
        raise ValueError("Boundary error")
    angle = get_angle(boundary_darts[0], boundary_darts[1], n)
    return angle


def on_boundary(n: Node) -> bool:
    """
    Test if the node n is on boundary.
    :param n: a node in the mesh.
    :return: True if the node n is on boundary, False otherwise.
    """
    adj_darts_list = adjacent_darts(n)
    for d in adj_darts_list:
        d_twin = d.get_beta(2)
        if d_twin is None:
            return True
    return False


def adjacent_darts(n: Node) -> list[Dart]:
    """
    Function that retrieve the adjacent darts of node n.
    :param n: a node in the mesh.
    :return: the list of adjacent darts
    """
    adj_darts = []
    for d_info in n.mesh.dart_info:
        d = Dart(n.mesh, d_info[0])
        d_nfrom = d.get_node()
        d_nto = d.get_beta(1)
        if d_nfrom == n:
            adj_darts.append(d)
        if d_nto.get_node() == n:
            adj_darts.append(d)
    return adj_darts


def degree(n: Node) -> int:
    """
    Function to calculate the degree of a node in the mesh.
    :param n: a node in the mesh.
    :return: the degree of the node
    """
    adj_darts_list = adjacent_darts(n)
    adjacency = 0
    b = on_boundary(n)
    boundary_darts = []
    for d in adj_darts_list:
        d_twin = d.get_beta(2)
        if d_twin is None and b:
            adjacency += 1
            boundary_darts.append(d)
        else:
            adjacency += 0.5

    return adjacency


def get_boundary_darts(m: Mesh) -> list[Dart]:
    boundary_darts = []
    for d_info in m.dart_info:
        d = Dart(m, d_info[0])
        d_twin = d.get_beta(2)
        if d_twin is None :
            boundary_darts.append(d)
    return boundary_darts


def get_boundary_nodes(m: Mesh) -> list[Node]:
    boundary_nodes = []
    nb_nodes = len(m.nodes)
    for n_id in range(0, nb_nodes):
        n = Node(m, n_id)
        if on_boundary(n):
            boundary_nodes.append(n)
    return boundary_nodes
