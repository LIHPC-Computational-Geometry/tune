from math import sqrt, degrees, radians, cos, sin, acos
import numpy as np

from model.mesh_struct.mesh_elements import Dart, Node, Face
from model.mesh_struct.mesh import Mesh


def global_score(m: Mesh):
    """
    Calculate the overall mesh score. The mesh cannot achieve a better score than the ideal one.
    And the current score is the mesh score.
    :param m: the mesh to be analyzed
    :return: three return values: a list of the nodes score, the current mesh score and the ideal mesh score
    """
    mesh_ideal_score = 0
    mesh_score = 0
    nodes_score = []
    active_nodes_score = []
    for i in range(len(m.nodes)):
        if m.nodes[i, 2] >= 0:
            n_id = i
            node = Node(m, n_id)
            n_score = score_calculation(node)
            nodes_score.append(n_score)
            active_nodes_score.append(n_score)
            mesh_ideal_score += n_score
            mesh_score += abs(n_score)
        else:
            nodes_score.append(0)
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
    cos_theta = np.dot(vect_AB, vect_AC)/(dist_AB*dist_AC)
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)
    if np.isnan(angle):
        raise(ValueError("Angle error"))
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
        print(adjacency)
        print(n.id)
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


def get_boundary_nodes(m: Mesh) -> list[Node]:
    """
    Find all boundary nodes
    :param m: a mesh
    :return: a list of all boundary nodes
    """
    boundary_nodes = []
    for n_id in range(0, len(m.nodes)):
        if m.nodes[n_id, 2] >= 0:
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


def find_template_opposite_node(d: Dart) -> int:
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
        if n[2] >= 0 :
            if abs(x - n[0]) <= 0.1 and abs(y - n[1]) <= 0.1:
                return True, n_id
        n_id += 1
    return False, None


def isValidAction(mesh: Mesh, dart_id: int, action: int) -> bool:
    flip = 0
    split = 1
    collapse = 2
    test_all = 3
    d = Dart(mesh, dart_id)
    boundary_darts = get_boundary_darts(mesh)
    if d in boundary_darts:
        return False
    elif action == flip:
        return isFlipOk(d)
    elif action == split:
        return isFlipOk(d)
    elif action == collapse:
        return newIsCollapseOk(d)
    elif action == test_all:
        return isFlipOk(d) and newIsCollapseOk(d)
    else:
        raise ValueError("No valid action")


def get_angle_by_coord(x1: float, y1: float, x2: float, y2: float, x3:float, y3:float) -> float:
    BAx, BAy = x1 - x2, y1 - y2
    BCx, BCy = x3 - x2, y3 - y2

    cos_ABC = (BAx * BCx + BAy * BCy) / (sqrt(BAx ** 2 + BAy ** 2) * sqrt(BCx ** 2 + BCy ** 2))
    cos_ABC = np.clip(cos_ABC, -1, 1)
    rad = acos(cos_ABC)
    deg = degrees(rad)
    return deg


def isFlipOk(d: Dart) -> bool:
    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    A = d.get_node()
    B = d1.get_node()
    C = d11.get_node()
    d2 = d.get_beta(2)
    if d2 is None:
        return False
    else:
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


def newIsCollapseOk(d: Dart) -> bool:
    mesh = d.mesh
    d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)

    d112 = d11.get_beta(2)
    d12 = d1.get_beta(2)

    d212 = d21.get_beta(2)
    d2112 = d211.get_beta(2)

    newNode_x, newNode_y = (n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2

    if d112 is None or d12 is None or d2112 is None or d212 is None:
        return False
    elif on_boundary(n1) or on_boundary(n2):
        return False
    else:
        # search for all adjacent faces to n1 and n2
        if d12 is None and d2112 is None:
            adj_faces_n1 = get_adjacent_faces(n1, d212, d112)
            return valid_faces_changes(adj_faces_n1, n1.id, newNode_x, newNode_y)
        elif d212 is None and d112 is None:
            adj_faces_n2 = get_adjacent_faces(n2, d12, d2112)
            return valid_faces_changes(adj_faces_n2, n2.id, newNode_x, newNode_y)
        else:
            adj_faces_n1 = get_adjacent_faces(n1, d212, d112)
            adj_faces_n2 = get_adjacent_faces(n2, d12, d2112)
            if not valid_faces_changes(adj_faces_n1, n1.id, newNode_x, newNode_y) or not valid_faces_changes(adj_faces_n2, n2.id, newNode_x, newNode_y):
                return False
            else:
                return True


def get_adjacent_faces(n: Node, d_from: Dart, d_to: Dart) -> list:
    adj_faces = []
    d2 = d_from
    d = None if d2 is None else d_from.get_beta(1)
    while d != d_to:
        if d2 is None and d_to is not None:
            # chercher dans l'autre sens
            d = d_to
            adj_faces.append(d.get_face())
            d1 = d.get_beta(1)
            d11 = d1.get_beta(1)
            d = d11.get_beta(2)
            while d is not None:
                adj_faces.append(d.get_face())
                d1 = d.get_beta(1)
                d11 = d1.get_beta(1)
                d = d11.get_beta(2)
            break
        elif d2 is None and d_to is None:
            break
        elif d2 is not None:
            d = d2.get_beta(1)
            adj_faces.append(d.get_face())
            d2 = d.get_beta(2)
        else:
            break
    return adj_faces

def discontinue(d_from, d_to) -> bool:
    if d_from is None or d_to is None:
        raise ValueError("Discontinue condition")

    ds = d_from.get_beta(1)
    while ds != d_to:
        ds2 = ds.get_beta(2)
        if ds2 is None:
            return True
        ds = ds2.get_beta(1)
    d1 = d_from.get_beta(1)
    ds = d1.get_beta(1)
    i = 0
    while ds != d_to:
        ds2 = ds.get_beta(2)
        if ds2 is None or i > 10:
            return True
        ds21 = ds2.get_beta(1)
        ds = ds21.get_beta(1)
        i += 1
    return False

def valid_faces_changes(faces: list[Face], n_id: int, new_x: float, new_y: float) -> bool:
    """
    Check the orientation of triangles adjacent to node n = Node(mesh, n_id) if the latter is moved to coordinates new_x, new_y.
    Also checks that no triangle will become flat
    :param mesh:
    :param faces: adjacents faces to node of id n_id
    :param n_id:
    :param new_x:
    :param new_y:
    :return:
    """
    for f in faces:
        d, d1, d11, A, B, C = f.get_surrounding()
        if A.id == n_id:
            vect_AB = (B.x() - new_x, B.y() - new_y)
            vect_AC = (C.x() - new_x, C.y() - new_y)
        elif B.id == n_id:
            vect_AB = (new_x - A.x(), new_y - A.y())
            vect_AC = (C.x() - A.x(), C.y() - A.y())
        elif C.id == n_id:
            vect_AB = (B.x() - A.x(), B.y() - A.y())
            vect_AC = (new_x - A.x(), new_y - A.y())
        else:
            print("Erreur face non adjacente")
            continue

        cross_product = vect_AB[0] * vect_AC[1] - vect_AB[1] * vect_AC[0]

        if cross_product <= 0:
            return False  # Une face n'est pas orientée correctement ou est plate
    return True

    """
    elif d112 is not None and d212 is not None:
        if d12 is None and d2112 is None:
            # search for discontinuities
            ds = d212.get_beta(1)
            while ds != d112:
                ds2 = ds.get_beta(2)
                if ds2 is None:
                    return False
                ds = ds2.get_beta(1)
            return True
    elif d12 is not None:
        # search for discontinuities
        ds = d12.get_beta(1)
        while ds != d2112:
            ds2 = ds.get_beta(2)
            if ds2 is None:
                return False
            ds = ds2.get_beta(1)
        return True
    else:
        return False

    #Old condition
    if d112 is None and d12 is None:
        return False
    elif d212 is None and d2112 is None:
        return False
    elif d212 is None and d12 is None:
        return False
    elif d112 is None and d2112 is None:
        return False
    else:
        return True


def isCollapseOk(d: Dart) -> bool:
    mesh = d.mesh
    d2, d1, d11, d21, d211, n1, n2, n3, n4 = mesh.active_triangles(d)

    d112 = d11.get_beta(2)
    d12 = d1.get_beta(2)

    d212 = d21.get_beta(2)
    d2112 = d211.get_beta(2)

    if d112 is None or d12 is None or d2112 is None or d212 is None:
        return False
    else:
        # search for discontinuities on right side (d12 and d2112)
        if discontinue(d12, d2112):
            return False
        if discontinue(d212, d112):
            return False
        else:
            return True
    """