import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt, degrees, acos, atan2
from abc import ABC, abstractmethod
from shapely import affinity
from shapely.geometry import Polygon, Point, LineString

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from view.mesh_plotter.mesh_plots import plot_mesh


class NodeAnalysis:
    def __init__(self, n: Node):
        self.n = n

    def score_calculation(self) -> int:
        """
        Function to calculate the irregularity of a node in the mesh.
        :param n: a node in the mesh.
        :return: the irregularity of the node
        :raises ValueError: if the node is associated to no dart
        """

        d = self.n.get_dart()
        if d.mesh.dart_info[d.id,0] < 0:
            raise ValueError("No existing dart")

        adjacency = self.degree()
        ideal_adjacency =self.n.get_ideal_adjacency()

        return ideal_adjacency - adjacency

    def get_angle(self, d1: Dart, d2: Dart) -> float:
        """
        Function to calculate the angle of the boundary at the node n.
        The angle is named ABC and node self.n is at point A.
        :param d1: the first boundary dart.
        :param d2: the second boundary dart.
        :return: the angle (degrees)
        """
        if d1.get_node() == self.n:
            A = self.n
            B = d1.get_beta(1).get_node()
            C = d2.get_node()

        else:
            A = self.n
            B = d2.get_beta(1).get_node()
            C = d1.get_node()
            if d2.get_node() != A:
                raise ValueError("Angle error")

        vect_AB = (B.x() - A.x(), B.y() - A.y())
        vect_AC = (C.x() - A.x(), C.y() - A.y())

        dot = vect_AB[0] * vect_AC[0] + vect_AB[1] * vect_AC[1]

        # cross product
        cross = vect_AB[0] * vect_AC[1] - vect_AB[1] * vect_AC[0]

        angle_rad = atan2(cross, dot)
        angle = degrees(angle_rad) % 360
        if np.isnan(angle):
            raise ValueError("Angle error")
        return angle

    def get_boundary_angle(self) -> float:
        """
        Calculate the boundary angle of a node in the mesh.
        :return: the boundary angle (degrees)
        """
        adj_darts_list = self.adjacent_darts()
        boundary_darts = []
        for d in adj_darts_list:
            d_twin = d.get_beta(2)
            if d_twin is None:
                boundary_darts.append(d)
        #if len(boundary_darts) > 2 : # or len(boundary_darts) < 2
            #plot_mesh(self.n.mesh)
            #raise ValueError("Boundary error")
        angle = self.get_angle(boundary_darts[0], boundary_darts[1])
        return angle

    def on_boundary(self) -> bool:
        """
        Test if the node self.n is on boundary.
        :return: True if the node n is on boundary, False otherwise.
        """
        adj_darts_list = self.adjacent_darts()
        for d in adj_darts_list:
            d_twin = d.get_beta(2)
            if d_twin is None:
                return True
        return False

    def adjacent_darts(self) -> list[Dart]:
        """
        Function that retrieve the adjacent darts of node n.
        :return: the list of adjacent darts
        """
        adj_darts = []
        m = self.n.mesh
        for d_info in m.active_darts():
            d = Dart(m, d_info[0])
            d_nfrom = d.get_node()
            d_nto = d.get_beta(1)
            if d_nfrom == self.n and d not in adj_darts:
                adj_darts.append(d)
            if d_nto.get_node() == self.n and d not in adj_darts:
                adj_darts.append(d)
        return adj_darts

    def adjacent_faces_id(self) -> list[int]:
        adj_darts = self.adjacent_darts()
        adj_faces = []
        for d in adj_darts:
            f = d.get_face()
            if f.id not in adj_faces:
                adj_faces.append(f.id)
        return adj_faces

    def degree(self) -> int:
        """
        Function to calculate the degree of a node in the mesh.
        :return: the degree of the node
        """
        adj_darts_list = self.adjacent_darts()
        adjacency = 0
        b = self.on_boundary()
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

    def test_degree(self) -> bool:
        """
        Verify that the degree of a vertex is lower than 10
        :return: True if the degree is lower than 10, False otherwise
        """
        if self.degree() >= 10:
            return False
        else:
            return True


class GlobalMeshAnalysis(ABC):
    """
    The base of mesh analysis
    :param mesh: A mesh to analise
    """
    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    def set_adjacency(self):
        pass

    def set_scores(self):
        pass

    def set_geometric_quality(self):
        pass

    def get_dart_geometric_quality(self, d: Dart) -> int:
        pass

    def get_dart_kernel(self, d: Dart, plot=False) -> (int, float, float):
        """
        Calculate the kernel of geometry around the dart. Return if the kernel exists, if so, return the coordinates of the kernel.
        :param d: the dart we want to calculate the kernel for
        :return: 1 if it has a kernel, -1 if next to boundary, 0 otherwise, x-coordinate, y-coordinate
        """
        d_init = d
        d1 = d.get_beta(1)
        d1_init=d1
        n1 = d.get_node()
        n2 = d1_init.get_node()
        na1 = NodeAnalysis(n1)
        na2 = NodeAnalysis(n2)


        if d.get_beta(2) is None:
            return -1, 0, 0
        elif na1.on_boundary() or na2.on_boundary():
            return -1, 0, 0

        d11 = d1.get_beta(1)
        d_to = d11
        adj_nodes = []
        nodes_coord = []

        d = d1.get_beta(2)
        #d12 = d1.get_beta(2)
        #d = d12.get_beta(1)
        n = d.get_node()

        while n != n1:
            adj_nodes.append(n)
            nodes_coord.append([n.x(), n.y()])
            d1 = d.get_beta(1)
            d = d1.get_beta(2)
            if d is None:
                raise ValueError("Error on getting kernel")
            n = d.get_node()

        d2 = d.get_beta(2)
        d212 = (d2.get_beta(1)).get_beta(2)
        d = (d212.get_beta(1)).get_beta(2)
        n = d.get_node()
        while d != d_to:
            adj_nodes.append(n)
            nodes_coord.append([n.x(), n.y()])
            d1 = d.get_beta(1)
            d = d1.get_beta(2)
            n = d.get_node()

        nodes_coord_tuple = []
        for coord in nodes_coord:
            tuple_coord = tuple(coord)
            nodes_coord_tuple.append(tuple_coord)
        if len(nodes_coord_tuple) != len(set(nodes_coord_tuple)):
            return 0, 0, 0

        nodes_coord.append(nodes_coord[0]) # We add the first point to close the ring
        nodes_coord = np.array(nodes_coord)

        if len(nodes_coord) <4:
            plot_mesh(d.mesh, debug=True)
            raise ValueError("oui")
        # Create a Polygon with shapely package
        poly = Polygon(nodes_coord)

        # If polygon is convexe
        if poly.is_valid and poly.is_simple and poly.convex_hull.equals(poly):
            centroid = poly.centroid
            return 1, centroid.x, centroid.y

        found, x, y = self.find_star_vertex_from_poly(poly, nodes_coord, plot=plot)

        if not found:
            return 0, 0, 0
        else:
            return 1, x, y

    def find_star_vertex_from_poly(self, poly: Polygon, nodes_coord, plot=False)-> (bool, float, float):
        #Else if polygon is concav, we're looking if there is a "star area"
        p_before = None
        p_first = None
        star_poly = poly
        for p in poly.exterior.coords[:-1]:
            if p_before is not None:
                # one side of the polygon
                seg = LineString([p, p_before])

                # we create a big quad box
                box =  Polygon([(-5, 0), (5, 0), (5, 5), (-5, 5), (-5, 0)])

                # segment angle
                dx = seg.coords[1][0] - seg.coords[0][0]
                dy = seg.coords[1][1] - seg.coords[0][1]
                angle = math.degrees(math.atan2(dy, dx))

                # moving box
                box_rot = affinity.rotate(box, angle, origin=(0, 0))
                origin_pt = Point(seg.coords[0])
                box_final = affinity.translate(box_rot, origin_pt.x, origin_pt.y)

                star_poly = star_poly.intersection(box_final)
                if star_poly.is_empty or not isinstance(star_poly, Polygon):
                    return False, 0, 0
                if plot:
                    plt.figure(figsize=(6, 6))
                    # Polygone
                    x, y = poly.exterior.xy
                    plt.fill(x, y, alpha=0.3, edgecolor='blue',
                             label='Polygon formed par by neighbours vertices')

                    # Voisins
                    plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')

                    # Polygone
                    x_s, y_s = box_final.exterior.xy
                    plt.fill(x_s, y_s, alpha=0.3, facecolor='lightcoral',
                             label='Star area')

                    plt.legend()
                    plt.gca().set_aspect('equal')
                    plt.show()
            elif p_before is None:
                p_first = p
            p_before = p

        #We do the same for the last segment
        seg = LineString([p_first, p_before])
        box = Polygon([(-5, 0), (5, 0), (5, 15), (-5, 5)])
        dx = seg.coords[1][0] - seg.coords[0][0]
        dy = seg.coords[1][1] - seg.coords[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        box_rot = affinity.rotate(box, angle, origin=(0, 0))
        origin_pt = Point(seg.coords[0])
        box_final = affinity.translate(box_rot, origin_pt.x, origin_pt.y)

        star_poly = star_poly.intersection(box_final)
        if star_poly.is_empty or not isinstance(star_poly, Polygon):
            return False, 0, 0
        if plot:
            plt.figure(figsize=(6, 6))
            # Polygone
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.3, edgecolor='blue',
                     label='Polygon formed par by neighbours vertices')

            # Voisins
            plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')

            # Polygone
            x_s, y_s = box_final.exterior.xy
            plt.fill(x_s, y_s, alpha=0.3, facecolor='lightcoral',
                     label='Star area')

            plt.legend()
            plt.gca().set_aspect('equal')
            plt.show()
        centroid = star_poly.centroid

        if plot:
            plt.figure(figsize=(6, 6))
            # Polygone
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.3, edgecolor='blue',
                     label='Polygon formed par by neighbours vertices')

            # Voisins
            plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')

            # Polygone
            x_s, y_s = star_poly.exterior.xy
            plt.fill(x_s, y_s, alpha=0.3, facecolor='lightcoral',
                     label='Star area')
            plt.scatter(centroid.x, centroid.y, color='red', marker='x', s=100, zorder=10, label='New point')
            plt.legend()
            plt.gca().set_aspect('equal')
            plt.show()

        return True, centroid.x ,centroid.y

    def isConvexPolygon(self, nodes_coord) -> int:

        nodes_coord = np.array(nodes_coord)

        # Create a Polygon with shapely package
        poly = Polygon(nodes_coord)

        # If polygon is convex
        if poly.is_valid and poly.is_simple and poly.convex_hull.equals(poly):
            return 1
        else:
            return 0
    def global_score(self):
        """
        Calculate the overall mesh score. The mesh cannot achieve a better score than the ideal one.
        And the current score is the mesh score.
        :return: 4 return: a list of the nodes score, the current mesh score and the ideal mesh score, and the adjacency
        """
        mesh_ideal_score = 0
        mesh_score = 0
        nodes_score = []
        nodes_adjacency = []
        for i in range(len(self.mesh.nodes)):
            if self.mesh.nodes[i, 2] >= 0:
                n_id = i
                node = Node(self.mesh, n_id)
                n_a = NodeAnalysis(node)
                n_score = n_a.score_calculation()
                nodes_score.append(n_score)
                nodes_adjacency.append(n_a.degree())
                mesh_ideal_score += n_score
                mesh_score += abs(n_score)
            else:
                nodes_score.append(0)
                nodes_adjacency.append(6)
        return nodes_score, mesh_score, mesh_ideal_score, nodes_adjacency

    def get_boundary_darts(self) -> list[Dart]:
        """
        Find all boundary darts
        :return: a list of all boundary darts
        """
        boundary_darts = []
        for d_info in self.mesh.active_darts():
            d = Dart(self.mesh, d_info[0])
            d_twin = d.get_beta(2)
            if d_twin is None:
                boundary_darts.append(d)
        return boundary_darts

    def node_in_mesh(self, x: float, y: float) -> (bool, int):
        """
        Search if the node of coordinate (x, y) is inside the mesh.
        :param x: X coordinate
        :param y: Y coordinate
        :return: a boolean indicating if the node is inside the mesh and the id of the node if it is.
        """
        n_id = 0
        for n in self.mesh.nodes:
            if n[2] >= 0:
                if abs(x - n[0]) <= 0.1 and abs(y - n[1]) <= 0.1:
                    return True, n_id
            n_id += 1
        return False, None

    def check_beta2_relation(self) -> bool:
        for dart_info in self.mesh.active_darts():
            d = dart_info[0]
            d2 = dart_info[2]
            if d2 >= 0 and self.mesh.dart_info[d2, 0] < 0:
                raise ValueError("error beta2")
            elif d2 >= 0 and self.mesh.dart_info[d2, 2] != d:
                raise ValueError("error beta2")
        return True

    def check_double(self) -> bool:
        for dart_info in self.mesh.active_darts():
            d = Dart(self.mesh, dart_info[0])
            d2 = Dart(self.mesh, dart_info[2]) if dart_info[2] >= 0 else None
            n1 = dart_info[3]
            if d2 is None:
                d1 = d.get_beta(1)
                n2 = d1.get_node().id
            else:
                n2 = d2.get_node().id
            for dart_info2 in self.mesh.active_darts():
                ds = Dart(self.mesh, dart_info2[0])
                ds2 = Dart(self.mesh, dart_info2[2]) if dart_info2[2] >= 0 else None
                if d != ds and d != ds2:
                    ns1 = dart_info2[3]
                    if ds2 is None:
                        ds1 = ds.get_beta(1)
                        ns2 = ds1.get_node().id
                    else:
                        ns2 = ds2.get_node().id

                    if n1 == ns1 and n2 == ns2:
                        plot_mesh(self.mesh)
                        raise ValueError("double error")
                    elif n2 == ns1 and n1 == ns2:
                        plot_mesh(self.mesh)
                        raise ValueError("double error")
        return True

    def mesh_check(self) -> bool:
        return self.check_double() and self.check_beta2_relation()

    def angle_from_sides(self, a, b, c):
        # Calculate angle A, with a the opposite side and b and c the adjacent sides
        cosA = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
        if 1 <= cosA < 1.01:
            cosA = 1
        elif -1.01 <= cosA < -1:
            cosA = -1
        elif cosA > 1.01 or cosA < -1.01:
            raise ValueError("Math domain error : cos>1.01")
        return acos(cosA)

    def get_angle_by_coord(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
        BAx, BAy = x1 - x2, y1 - y2
        BCx, BCy = x3 - x2, y3 - y2

        cos_ABC = (BAx * BCx + BAy * BCy) / (sqrt(BAx ** 2 + BAy ** 2) * sqrt(BCx ** 2 + BCy ** 2))
        cos_ABC = np.clip(cos_ABC, -1, 1)
        rad = acos(cos_ABC)
        deg = degrees(rad)
        return deg

    def cross_product(self, vect_AB, vect_AC):
        """ Return the cross product between AB et AC.
            0 means A, B and C are coolinear
            > 0 mean A, B and C are "sens des aiguilles d'une montre"
            < 0 sens inverse
        """
        val = vect_AB[0] * vect_AC[1] - vect_AB[1] * vect_AC[0]
        return val

    def signe(self, a: int):
        pass