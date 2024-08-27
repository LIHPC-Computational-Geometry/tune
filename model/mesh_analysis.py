from math import sqrt, degrees, radians, cos, sin, acos
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
    angle = np.arccos(np.dot(vect_AB, vect_AC)/(dist_AB*dist_AC))
    return degrees(angle)


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
        if d_nfrom == n and d not in adj_darts:
            adj_darts.append(d)
        if d_nto.get_node() == n and d not in adj_darts:
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
    for d_info in m.dart_info:
        d = Dart(m, d_info[0])
        d_twin = d.get_beta(2)
        if d_twin is None :
            boundary_darts.append(d)
    return boundary_darts


def get_boundary_nodes(m: Mesh) -> list[Node]:
    """
    Find all boundary nodes
    :param m: a mesh
    :return: a list of all boundary nodes
    """
    boundary_nodes = []
    nb_nodes = len(m.nodes)
    for n_id in range(0, nb_nodes):
        n = Node(m, n_id)
        if on_boundary(n):
            boundary_nodes.append(n)
    return boundary_nodes


def find_opposite_node(d: Dart) -> (int, int):
    """
    Find the coordinates of the vertex opposite in the adjacent triangle
    :param d: a dart
    :return: (X Coordinate, Y Coordinate)
    """
    A = d.get_node()
    d1 = d.get_beta(1)
    B = d1.get_node()

    vect_AB = (B.x() - A.x(), B.y() - A.y())

    angle_rot = radians(300)
    x_AC = round(vect_AB[0] * cos(angle_rot) - vect_AB[1] * sin(angle_rot), 2)
    y_AC = round(vect_AB[1] * cos(angle_rot) + vect_AB[0] * sin(angle_rot), 2)

    x_C = A.x() + x_AC
    y_C = A.y() + y_AC

    return x_C, y_C

def find_template_opposite_node(d: Dart) -> (int):
    """
    Find the the vertex opposite in the adjacent triangle
    :param d: a dart
    :return: the node found
    """

    d2 = d.get_beta(2)
    if d2 is not None:
        d21 = d2.get_beta(1)
        d211 = d21.get_beta(1)
        node_opposite = d211.get_node()
        return node_opposite.id
    else:
        return None


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
        if abs(x - n[0]) <= 0.1 and abs(y - n[1]) <= 0.1:
            return True, n_id
        n_id = n_id + 1
    return False, None


def isValidAction(mesh, dart_id: int) -> bool:
    d = Dart(mesh, dart_id)
    boundary_darts = get_boundary_darts(mesh)
    if d in boundary_darts or not isFlipOk(d):
        return False
    else:
        return True

def get_angle_by_coord(x1: float, y1: float, x2: float, y2: float, x3:float, y3:float) -> float:
    BAx, BAy = x1 - x2, y1 - y2
    BCx, BCy = x3 - x2, y3 - y2

    cos_ABC = (BAx * BCx + BAy * BCy) / (sqrt(BAx ** 2 + BAy ** 2) * sqrt(BCx ** 2 + BCy ** 2))

    rad = acos(cos_ABC)
    deg = degrees(rad)
    return deg


def isFlipOk(d:Dart) -> bool:
    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    A = d.get_node()
    B = d1.get_node()
    C = d11.get_node()
    d2 = d.get_beta(2)
    d21 = d2.get_beta(1)
    d211 = d21.get_beta(1)
    D = d211.get_node()

    # Calcul angle at d limits
    angle_B = get_angle_by_coord(A.x(), A.y(), B.x(), B.y(), C.x(), C.y()) + get_angle_by_coord(A.x(), A.y(), B.x(), B.y(), D.x(), D.y())
    angle_A = get_angle_by_coord(B.x(), B.y(), A.x(), A.y(), C.x(), C.y()) + get_angle_by_coord(B.x(), B.y(), A.x(), A.y(), D.x(), D.y())

    if angle_B >= 180 or angle_A >= 180:
        return False
    else:
        return True
