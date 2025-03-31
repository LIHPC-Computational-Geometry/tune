from math import sqrt, degrees, acos
import numpy as np

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from view.mesh_plotter.mesh_plots import plot_mesh


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
    # Check if the mesh is triangular or quad
    d = n.get_dart()
    if d.id <0:
        raise ValueError("No existing dart")
    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    d111 = d11.get_beta(1)
    triangular = (d111.id == d.id)
    if on_boundary(n):
        angle = get_boundary_angle(n)
        if triangular:
            ideal_adjacency = max(round(angle/60)+1, 2)
        else:
            ideal_adjacency = max(round(angle/90)+1, 2)
    elif triangular:
        ideal_adjacency = 360/60
    else:
        ideal_adjacency = 360/90

    return ideal_adjacency-adjacency, adjacency

def get_angle(d1: Dart, d2: Dart, n: Node) -> float:
    """
    Function to calculate the angle of the boundary at the node n.
    The angle is named ABC and node n is at point A.
    :param d1: the first boundary dart.
    :param d2: the second boundary dart.
    :param n: the boundary node
    :return: the angle (degrees)
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
    cos_theta = np.dot(vect_AB, vect_AC)/(dist_AB*dist_AC)
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)
    if np.isnan(angle):
        plot_mesh(n.mesh)
        raise ValueError("Angle error")
    return degrees(angle)

def angle_from_sides(a, b, c):
    # Calculate angle A, with a the opposite side and b and c the adjacent sides
    cosA = (b**2 + c**2 - a**2) / (2 * b * c)
    if 1 <= cosA < 1.01:
        cosA = 1
    elif -1.01 <= cosA < -1:
        cosA = -1
    elif cosA > 1.01 or cosA < -1.01:
        raise ValueError("Math domain error : cos>1.01")
    return acos(cosA)

def get_angle_by_coord(x1: float, y1: float, x2: float, y2: float, x3:float, y3:float) -> float:
    BAx, BAy = x1 - x2, y1 - y2
    BCx, BCy = x3 - x2, y3 - y2

    cos_ABC = (BAx * BCx + BAy * BCy) / (sqrt(BAx ** 2 + BAy ** 2) * sqrt(BCx ** 2 + BCy ** 2))
    cos_ABC = np.clip(cos_ABC, -1, 1)
    rad = acos(cos_ABC)
    deg = degrees(rad)
    return deg

def get_boundary_angle(n: Node) -> float:
    """
    Calculate the boundary angle of a node in the mesh.
    :param n: a boundary node
    :return: the boundary angle (degrees)
    """
    adj_darts_list = adjacent_darts(n)
    boundary_darts = []
    for d in adj_darts_list:
        d_twin = d.get_beta(2)
        if d_twin is None:
            boundary_darts.append(d)
    if len(boundary_darts) > 7:
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
    for d_info in n.mesh.active_darts():
        d = Dart(n.mesh, d_info[0])
        d_nfrom = d.get_node()
        d_nto = d.get_beta(1)
        if d_nfrom == n and d not in adj_darts:
            adj_darts.append(d)
        if d_nto.get_node() == n and d not in adj_darts:
            adj_darts.append(d)
    return adj_darts

def adjacent_faces(n: Node) -> list[Face]:
    adj_darts = adjacent_darts(n)
    adj_faces = []
    for d in adj_darts:
        f = d.get_face()
        if f not in adj_faces:
            adj_faces.append(f)
    return adj_faces


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
    if adjacency != int(adjacency):
        raise ValueError("Adjacency error")
    return adjacency


def get_boundary_darts(m: Mesh) -> list[Dart]:
    """
    Find all boundary darts
    :param m: a mesh
    :return: a list of all boundary darts
    """
    boundary_darts = []
    for d_info in m.active_darts():
        d = Dart(m, d_info[0])
        d_twin = d.get_beta(2)
        if d_twin is None:
            boundary_darts.append(d)
    return boundary_darts


def node_in_mesh(mesh: Mesh, x: float, y: float) -> (bool, int):
    """
    Search if the node of coordinate (x, y) is inside the mesh.
    :param mesh: the mesh to work with
    :param x: X coordinate
    :param y: Y coordinate
    :return: a boolean indicating if the node is inside the mesh and the id of the node if it is.
    """
    n_id = 0
    for n in mesh.nodes:
        if n[2] >= 0 :
            if abs(x - n[0]) <= 0.1 and abs(y - n[1]) <= 0.1:
                return True, n_id
        n_id += 1
    return False, None


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


def check_beta2_relation(mesh: Mesh) -> bool:
    for dart_info in mesh.active_darts():
        d = dart_info[0]
        d2 = dart_info[2]
        if d2 >= 0 and mesh.dart_info[d2, 0] < 0:
            raise ValueError("error beta2")
        elif d2 >= 0 and mesh.dart_info[d2, 2] != d:
            raise ValueError("error beta2")
    return True


def check_double(mesh: Mesh) -> bool:
    for dart_info in mesh.active_darts():
        d = Dart(mesh, dart_info[0])
        d2 = Dart(mesh, dart_info[2]) if dart_info[2] >= 0 else None
        n1 = dart_info[3]
        if d2 is None:
            d1 = d.get_beta(1)
            n2 = d1.get_node().id
        else:
            n2 = d2.get_node().id
        for dart_info2 in mesh.active_darts():
            ds = Dart(mesh, dart_info2[0])
            ds2 = Dart(mesh, dart_info2[2]) if dart_info2[2] >= 0 else None
            if d != ds and d != ds2:
                ns1 = dart_info2[3]
                if ds2 is None:
                    ds1 = ds.get_beta(1)
                    ns2 = ds1.get_node().id
                else:
                    ns2 = ds2.get_node().id

                if n1 == ns1 and n2 == ns2:
                    plot_mesh(mesh)
                    raise ValueError("double error")
                elif n2 == ns1 and n1 == ns2:
                    plot_mesh(mesh)
                    raise ValueError("double error")
    return True


def mesh_check(mesh: Mesh) -> bool:
    return check_double(mesh) and check_beta2_relation(mesh)


"""
def get_boundary_nodes(m: Mesh) -> list[Node]:
    #
    Find all boundary nodes
    :param m: a mesh
    :return: a list of all boundary nodes
    #
    boundary_nodes = []
    for n_id in range(0, len(m.nodes)):
        if m.nodes[n_id, 2] >= 0:
            n = Node(m, n_id)
            if on_boundary(n):
                boundary_nodes.append(n)
    return boundary_nodes
"""