import math
from math import radians, cos, sin

from scipy.spatial import ConvexHull, Delaunay
from shapely import affinity

from shapely.geometry import Polygon, Point, LineString

import numpy as np
import matplotlib.pyplot as plt

from mesh_model.mesh_analysis.global_mesh_analysis import GlobalMeshAnalysis, NodeAnalysis
from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from view.mesh_plotter.mesh_plots import plot_mesh

FLIP = 0
SPLIT = 1
COLLAPSE = 2
TEST_ALL = 3
ONE_VALID = 4

class TriMeshAnalysis(GlobalMeshAnalysis):
    """
    The base of triangular mesh analysis
    """
    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)
        #If initial scores and adjacency have not already been set
        if self.mesh.dart_info[0,5] == -99:
            self.set_adjacency()
            self.set_scores()
            self.set_geometric_quality()

    def isValidAction(self, dart_id: int, action: int) -> (bool,bool):
        pass

    def set_adjacency(self) -> None:
        i = 0
        for n_info in self.mesh.nodes:
            if n_info[2] >= 0:
                n = Node(self.mesh, i)
                na = NodeAnalysis(n)
                if na.on_boundary():
                    angle = na.get_boundary_angle()
                    ideal_adj = max(round(angle / 60) + 1, 2)
                    n.set_ideal_adjacency(ideal_adj)
                else:
                    n.set_ideal_adjacency(6)
            i += 1

    def set_scores(self) -> None:
        i = 0
        for n_info in self.mesh.nodes:
            if n_info[2] >= 0:
                n = Node(self.mesh, i)
                na = NodeAnalysis(n)
                s = na.score_calculation()
                n.set_score(s)
            i += 1

    def set_geometric_quality(self) -> None:
        for d_info in self.mesh.active_darts():
            d_id = d_info[0]
            d = Dart(self.mesh, d_id)
            d.set_quality(self.get_dart_geometric_quality(d))

    def get_dart_geometric_quality(self, d: Dart, m=None) -> int:
        """
        Calculate the geometric quality of the surrounding of a dart and his twin dart.
            * quality = -1: boundary dart
            * quality = 0: convex quad
            * quality = 1: concave quad
            * quality = 2: triangular quad
            * quality = 3: crossed quad
            * quality = 4: half flat concave quad
            * quality = 6: flat quad
            * quality = 5: one adjacent triangular face is flat

        :param d: dart
        :return: geometric quality
        """
        if d.get_beta(2) is None:
            return -1 # boundary dart

        d2, d1, d11, d21, d211, A, B, C, D = self.mesh.active_triangles(d)

        u1 = np.array([B.x() - A.x(), B.y() - A.y()])  # d vector (unused)
        u2 = np.array([C.x() - B.x(), C.y() - B.y()])  # vect(BC)
        u3 = np.array([A.x() - C.x(), A.y() - C.y()])  # vect(CA)
        u4 = np.array([D.x() - A.x(), D.y() - A.y()])  # vect(AD)
        u5 = np.array([B.x() - D.x(), B.y() - D.y()])  # vect(DB)

        # Checking for near-zero vectors (close to (0,0))
        if (np.linalg.norm(u2) < 1e-8 or
                np.linalg.norm(u3) < 1e-8 or
                np.linalg.norm(u4) < 1e-8 or
                np.linalg.norm(u5) < 1e-8):
            plot_mesh(self.mesh)
            #raise ValueError("near zero vector") # Quad invalid because one side is almost zero

        # Calculate cross product at each node
        cp_A = self.cross_product(-1 * u3, u4)
        cp_D = self.cross_product(-1 * u4, u5)
        cp_B = self.cross_product(-1 * u5, u2)
        cp_C = self.cross_product(-1 * u2, u3)

        # Counts how many angles are oriented clockwise. If the angle is clockwise oriented, signe(cp) return 1
        sum_cp = self.signe(cp_A) + self.signe(cp_B) + self.signe(cp_C) + self.signe(cp_D)


        zero_count = sum(-1e-8 < cp < 1e-8 for cp in [cp_A, cp_B, cp_C, cp_D])

        if zero_count == 0: # not angles of 180° or 360° / no coolinear vectors
            if sum_cp == 0:
                return 0 #convexe
            elif sum_cp == 1:
                return 1 # concave
            elif sum_cp == 2:
                return 3 # crossed
        elif zero_count == 1:
            if sum_cp == 0:
                return 2 # triangular quad
            elif sum_cp == 1:
                print(cp_A, cp_B, cp_C, cp_D)
                return 4 # half_face_flat
            elif sum_cp == 2:
                return 5 # half face flat
        elif zero_count >= 2:
            return 6 # full flat face
        else:
            raise ValueError("Quality configuration doesn't exist")

    def find_star_vertex(self, n:Node, plot=False)-> (bool, float, float):
        # Retrieve all neighboring vertices in order
        d = n.get_dart()
        d2 = d.get_beta(2)
        n_start = d2.get_node()  # First neighbour to retrieve

        adj_nodes = [n_start]
        nodes_coord = [[n_start.x(), n_start.y()]]

        d = d2.get_beta(1)
        d2 = d.get_beta(2)
        n_neighbour = d2.get_node()

        # As long as we haven't returned to the first neighbor, we keep searching.
        # This works because the collapse action is restricted to inner darts that are not connected to a boundary node.
        # Therefore, we are guaranteed to find the first vertex by following the beta1 and beta2 relations.

        while n_neighbour != n_start:
            adj_nodes.append(n_neighbour)
            nodes_coord.append([n_neighbour.x(), n_neighbour.y()])
            d = d2.get_beta(1)
            d2 = d.get_beta(2)
            n_neighbour = d2.get_node()

        nodes_coord = np.array(nodes_coord)

        # Create a Polygon with shapely package
        poly = Polygon(nodes_coord)

        # If polygon is convexe
        if poly.is_valid and poly.is_simple and poly.convex_hull.equals(poly):
            centroid = poly.centroid
            return True, centroid.x, centroid.y

        #Else if polygon is concav, we're looking if there is a "star area"
        p_before = None
        p_first = None
        star_poly = poly
        for p in poly.exterior.coords[:-1]:
            if p_before is not None:
                # one side of the polygon
                seg = LineString([p, p_before])

                # we create a big quad box
                box =  Polygon([(-1e5, 0), (1e5, 0), (1e5, 1e5), (-1e5, 1e5)])

                # segment angle
                dx = seg.coords[1][0] - seg.coords[0][0]
                dy = seg.coords[1][1] - seg.coords[0][1]
                angle = math.degrees(math.atan2(dy, dx))

                #moving box
                box_rot = affinity.rotate(box, angle, origin=(0, 0))
                origin_pt = Point(seg.coords[0])
                box_final = affinity.translate(box_rot, origin_pt.x, origin_pt.y)

                star_poly = star_poly.intersection(box_final)
                if star_poly.is_empty:
                    plt.figure(figsize=(6, 6))
                    # Polygone
                    x, y = poly.exterior.xy
                    plt.fill(x, y, alpha=0.3, edgecolor='blue',
                             label='Polygon formed par by neighbours vertices')

                    # Voisins
                    plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')

                    plt.legend()
                    plt.gca().set_aspect('equal')
                    plt.show()
                    return False, 0, 0
            elif p_before is None:
                p_first = p
            p_before = p

        #We do the same for the last segment
        seg = LineString([p_first, p_before])
        box = Polygon([(-1e5, 0), (1e5, 0), (1e5, 1e5), (-1e5, 1e5)])
        dx = seg.coords[1][0] - seg.coords[0][0]
        dy = seg.coords[1][1] - seg.coords[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        box_rot = affinity.rotate(box, angle, origin=(0, 0))
        origin_pt = Point(seg.coords[0])
        box_final = affinity.translate(box_rot, origin_pt.x, origin_pt.y)

        star_poly = star_poly.intersection(box_final)
        if star_poly.is_empty:
            plt.figure(figsize=(6, 6))
            # Polygone
            x, y = poly.exterior.xy
            plt.fill(x, y, alpha=0.3, edgecolor='blue',
                     label='Polygon formed par by neighbours vertices')

            # Voisins
            plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')

            plt.legend()
            plt.gca().set_aspect('equal')
            plt.show()
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
            x_s, y_s = star_poly.exterior.xy
            plt.fill(x_s, y_s, alpha=0.3, facecolor='lightcoral',
                     label='Star area')

            plt.legend()
            plt.gca().set_aspect('equal')
            plt.show()

        centroid = star_poly.centroid
        return True, centroid.x ,centroid.y

    def get_adjacent_faces(self, n: Node, d_from: Dart, d_to: Dart) -> list:
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

    def find_opposite_node(self, d: Dart) -> (int, int):
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

    def find_template_opposite_node(self, d: Dart) -> int:
        """
        Find the vertex opposite in the adjacent triangle
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

    def signe(self, a: int):
        if a <= 1e-8: # We compare to 1e-8 and not 0; otherwise, the triangular configuration may not be detected.
            return 0
        else:
            return 1

class TriMeshQualityAnalysis(TriMeshAnalysis):
    """
    The base of triangular mesh analysis
    """
    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)

    def isValidAction(self, dart_id: int, action: int) -> (bool,bool):
        d = Dart(self.mesh, dart_id)
        #boundary_darts = self.get_boundary_darts()

        geo = True # The geometric validity is automatically set to True, it is not tested here

        if d.get_beta(2) is None: #if d in boundary_darts:
            return False, geo
        elif action == FLIP:
            return self.isFlipOk(d)
        elif action == SPLIT:
            return self.isSplitOk(d)
        elif action == COLLAPSE:
            return self.isCollapseOk(d)
        elif action == TEST_ALL:
            topo, geo = self.isFlipOk(d)
            if not (topo and geo):
                return False, False
            topo, geo = self.isSplitOk(d)
            if not (topo and geo):
                return False, False
            topo, geo = self.isCollapseOk(d)
            if not (topo and geo):
                return False, False
            elif topo and geo:
                return True, True
        elif action == ONE_VALID:
            topo_flip, geo_flip = self.isFlipOk(d)
            if (topo_flip and geo_flip):
                return True, True
            topo_split, geo_split = self.isSplitOk(d)
            if (topo_split and geo_split):
                return True, True
            topo_collapse, geo_collapse = self.isCollapseOk(d)
            if (topo_collapse and geo_collapse):
                return True, True
            return False, False
        else:
            raise ValueError("No valid action")

    def isFlipOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True

        #if d is on boundary, flip is not possible
        if d.get_beta(2) is None:
            topo = False
            return topo, geo

        d2, d1, d11, d21, d211, A, B, C, D = self.mesh.active_triangles(d)

        nA_analysis = NodeAnalysis(A)
        nB_analysis = NodeAnalysis(B)

        if not nA_analysis.test_degree() or not nB_analysis.test_degree():
            topo = False

        #Edge reversal not allowed
        if d.get_quality() != 0:
            geo = False

        return topo, geo

    def isSplitOk(self, d: Dart) -> (bool,bool):
        topo = True
        geo = True
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        _, _, _, _, _, A, B, C, D = self.mesh.active_triangles(d)

        nC_analysis = NodeAnalysis(C)
        nD_analysis = NodeAnalysis(D)
        if not nC_analysis.test_degree() or not nD_analysis.test_degree():
            topo = False

        if d.get_quality() not in [0,1,2]: # if the face around is crossed or flat, or half flat
            geo = False

        return topo, geo

    def isCollapseOk(self, d: Dart) -> (bool,bool):

        topo = True
        geo = True

        if d.get_beta(2) is None:
            topo = False
            return topo, geo

        _, d1, d11, d21, d211, n1, n2, _, _ = self.mesh.active_triangles(d)

        d112 = d11.get_beta(2)
        d12 = d1.get_beta(2)

        d212 = d21.get_beta(2)
        d2112 = d211.get_beta(2)

        n1_analysis = NodeAnalysis(n1)
        n2_analysis = NodeAnalysis(n2)

        if n1_analysis.on_boundary() or n2_analysis.on_boundary():
            topo = False
        elif not n1_analysis.test_degree():
            topo = False
        elif d112 is None or d12 is None or d2112 is None or d212 is None:
            topo = False

        if d.get_quality() not in [0,1,2]:  # if the face around is crossed or flat, or half flat
            geo = False
        elif d1.get_quality() == 1 or d11.get_quality() == 1 or d21.get_quality() == 1 or d211.get_quality() == 1:
            geo = False
        # elif d.get_quality() == 1:
        #     n = d.get_node()
        #     na = NodeAnalysis(n)
        #     adj_darts = na.adjacent_darts()
        #     for d in adj_darts:
        #         if d.get_quality() == 1:
        #             geo = False
        return topo, geo

    def isTruncated(self, darts_list) -> bool:
        for d_id in darts_list:
            if self.isValidAction( d_id, 4)[0]:
                return False
        return True


class TriMeshOldAnalysis(TriMeshAnalysis):
    """
    Triangular mesh analysis with geometrical criteria
    """

    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)

    def isValidAction(self, dart_id: int, action: int) -> (bool, bool):
        d = Dart(self.mesh, dart_id)
        boundary_darts = self.get_boundary_darts()
        if d in boundary_darts:
            return False, True
        elif action == FLIP:
            return self.isFlipOk(d)
        elif action == SPLIT:
            return self.isSplitOk(d)
        elif action == COLLAPSE:
            return self.isCollapseOk(d)
        elif action == TEST_ALL:
            topo, geo = self.isFlipOk(d)
            if not (topo and geo):
                return False, False
            topo, geo = self.isSplitOk(d)
            if not (topo and geo):
                return False, False
            topo, geo = self.isCollapseOk(d)
            if not (topo and geo):
                return False, False
            elif topo and geo:
                return True, True
        elif action == ONE_VALID:
            topo_flip, geo_flip = self.isFlipOk(d)
            if (topo_flip and geo_flip):
                return True, True
            topo_split, geo_split = self.isSplitOk(d)
            if (topo_split and geo_split):
                return True, True
            topo_collapse, geo_collapse = self.isCollapseOk(d)
            if (topo_collapse and geo_collapse):
                return True, True
            return False, False
        else:
            raise ValueError("No valid action")

    def isFlipOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True
        # if d is on boundary, flip is not possible
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        else:
            _, _, _, _, _, A, B, C, D = self.mesh.active_triangles(d)

        nA_analysis = NodeAnalysis(A)
        nB_analysis = NodeAnalysis(B)

        if not nA_analysis.test_degree() or not nB_analysis.test_degree():
            topo = False
            return topo, geo

        # Check angle at d limits to avoid edge reversal
        angle_B = self.get_angle_by_coord(A.x(), A.y(), B.x(), B.y(), C.x(), C.y()) + self.get_angle_by_coord(A.x(),
                                                                                                              A.y(),
                                                                                                              B.x(),
                                                                                                              B.y(),
                                                                                                              D.x(),
                                                                                                              D.y())
        angle_A = self.get_angle_by_coord(B.x(), B.y(), A.x(), A.y(), C.x(), C.y()) + self.get_angle_by_coord(B.x(),
                                                                                                              B.y(),
                                                                                                              A.x(),
                                                                                                              A.y(),
                                                                                                              D.x(),
                                                                                                              D.y())
        if angle_B >= 180 or angle_A >= 180:
            topo = False
            return topo, geo

        # Check if new triangle will be valid

        # Triangle ACD
        vect_AC = (C.x() - A.x(), C.y() - A.y())
        vect_AD = (D.x() - A.x(), D.y() - A.y())
        vect_DC = (C.x() - D.x(), C.y() - D.y())

        # Triangle CBD
        vect_BC = (C.x() - B.x(), C.y() - B.y())
        vect_BD = (D.x() - B.x(), D.y() - B.y())

        if not self.valid_triangle(vect_AC, vect_AD, vect_DC) or not self.valid_triangle(vect_BC, vect_BD, vect_DC):
            geo = False
            return topo, geo

        return topo, geo

    def isSplitOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        else:
            _, _, _, _, _, A, B, C, D = self.mesh.active_triangles(d)

        nC_analysis = NodeAnalysis(C)
        nD_analysis = NodeAnalysis(D)

        if not nC_analysis.test_degree() or not nD_analysis.test_degree():
            topo = False
            return topo, geo

        newNode_x, newNode_y = (A.x() + B.x()) / 2, (A.y() + B.y()) / 2

        # Check if new triangle will be valid

        # Triangle AEC
        vect_AC = (C.x() - A.x(), C.y() - A.y())
        vect_AE = (newNode_x - A.x(), newNode_y - A.y())
        vect_EC = (C.x() - newNode_x, C.y() - newNode_y)
        if not self.valid_triangle(vect_AE, vect_AC, vect_EC):
            geo = False
            return topo, geo

        # Triangle ADE
        vect_AD = (D.x() - A.x(), D.y() - A.y())
        vect_ED = (D.x() - newNode_x, D.y() - newNode_y)
        if not self.valid_triangle(vect_AD, vect_AE, vect_ED):
            geo = False
            return topo, geo

        # Triangle BCE
        vect_BC = (C.x() - B.x(), C.y() - B.y())
        vect_BE = (newNode_x - B.x(), newNode_y - B.y())
        vect_EC = (C.x() - newNode_x, C.y() - newNode_y)
        if not self.valid_triangle(vect_BC, vect_BE, vect_EC):
            geo = False
            return topo, geo

        # Triangle BDE
        vect_BD = (D.x() - B.x(), D.y() - B.y())
        vect_ED = (D.x() - newNode_x, D.y() - newNode_y)
        if not self.valid_triangle(vect_BD, vect_BE, vect_ED):
            geo = False
            return topo, geo

        return topo, geo

    def isCollapseOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        else:
            _, d1, d11, d21, d211, n1, n2, _, _ = self.mesh.active_triangles(d)

        d112 = d11.get_beta(2)
        d12 = d1.get_beta(2)

        d212 = d21.get_beta(2)
        d2112 = d211.get_beta(2)

        newNode_x, newNode_y = (n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2

        n1_analysis = NodeAnalysis(n1)
        n2_analysis = NodeAnalysis(n2)

        if d112 is None or d12 is None or d2112 is None or d212 is None:
            topo = False
            return topo, geo
        elif n1_analysis.on_boundary() or n2_analysis.on_boundary():
            topo = False
            return topo, geo
        elif not n1_analysis.test_degree():
            topo = False
            return topo, geo
        else:
            # search for all adjacent faces to n1 and n2
            if d12 is None and d2112 is None:
                adj_faces_n1 = self.get_adjacent_faces(n1, d212, d112)
                return topo, self.valid_faces_changes(adj_faces_n1, n1.id, newNode_x, newNode_y)
            elif d212 is None and d112 is None:
                adj_faces_n2 = self.get_adjacent_faces(n2, d12, d2112)
                return topo, self.valid_faces_changes(adj_faces_n2, n2.id, newNode_x, newNode_y)
            else:
                adj_faces_n1 = self.get_adjacent_faces(n1, d212, d112)
                adj_faces_n2 = self.get_adjacent_faces(n2, d12, d2112)
                if not self.valid_faces_changes(adj_faces_n1, n1.id, newNode_x,
                                                newNode_y) or not self.valid_faces_changes(adj_faces_n2, n2.id,
                                                                                           newNode_x, newNode_y):
                    geo = False
                    return topo, geo
                else:
                    return topo, geo

    def valid_faces_changes(self, faces: list[Face], n_id: int, new_x: float, new_y: float) -> bool:
        """
        Check the orientation of triangles adjacent to node n = Node(mesh, n_id) if the latter is moved to coordinates new_x, new_y.
        Also checks that no triangle will become flat
        :param faces: adjacents faces to node of id n_id
        :param n_id: node id
        :param new_x: new x coordinate
        :param new_y: new y coordinate
        :return: True if valid, False otherwise
        """
        for f in faces:
            _, _, _, A, B, C = f.get_surrounding_triangle()
            if A.id == n_id:
                vect_AB = (B.x() - new_x, B.y() - new_y)
                vect_AC = (C.x() - new_x, C.y() - new_y)
                vect_BC = (C.x() - B.x(), C.y() - B.y())
            elif B.id == n_id:
                vect_AB = (new_x - A.x(), new_y - A.y())
                vect_AC = (C.x() - A.x(), C.y() - A.y())
                vect_BC = (C.x() - new_x, C.y() - new_y)
            elif C.id == n_id:
                vect_AB = (B.x() - A.x(), B.y() - A.y())
                vect_AC = (new_x - A.x(), new_y - A.y())
                vect_BC = (new_x - B.x(), new_y - B.y())
            else:
                print("Non-adjacent face error")
                continue

            cross_product = vect_AB[0] * vect_AC[1] - vect_AB[1] * vect_AC[0]

            if cross_product <= 0:
                return False  # One face is not correctly oriented or is flat
            elif not self.valid_triangle(vect_AB, vect_AC, vect_BC):
                return False
        return True

    def valid_triangle(self, vect_AB, vect_AC, vect_BC) -> bool:
        dist_AB = math.sqrt(vect_AB[0] ** 2 + vect_AB[1] ** 2)
        dist_AC = math.sqrt(vect_AC[0] ** 2 + vect_AC[1] ** 2)
        dist_BC = math.sqrt(vect_BC[0] ** 2 + vect_BC[1] ** 2)
        target_mesh_size = 1

        L_max = max(dist_AB, dist_AC, dist_BC)

        if target_mesh_size / 2 * math.sqrt(2) < L_max and L_max < target_mesh_size * 3 * math.sqrt(2):  # 0.35<Lmax<4.24
            pass
        else:
            return False

        # Calcul des angles avec le théorème du cosinus
        angle_B = math.degrees(self.angle_from_sides(dist_AC, dist_AB, dist_BC))  # Angle au point A
        angle_C = math.degrees(self.angle_from_sides(dist_AB, dist_BC, dist_AC))  # Angle au point B
        angle_A = math.degrees(self.angle_from_sides(dist_BC, dist_AC, dist_AB))  # Angle au point C

        # Vérification que tous les angles sont supérieurs à 5°
        if angle_A <= 5 or angle_B <= 5 or angle_C <= 5:
            return False
        return True

    def isTruncated(self, darts_list) -> bool:
        for d_id in darts_list:
            if self.isValidAction(d_id, 4)[0]: # if on action is valid, it means it's valid topologically and geometrically, so no need to verify the two
                return False
        return True


    # def is_star_vertex(self, n1:Node, new_coordinates, plot=False):
    #     #plot_mesh(self.mesh)
    #
    #     # Retrieve all neighboring vertices in order
    #     d = n1.get_dart()
    #     d2 = d.get_beta(2)
    #     n_start = d2.get_node() #First neighbour to retrieve
    #
    #     adj_nodes = [n_start]
    #     nodes_coord = [[n_start.x(), n_start.y()]]
    #
    #     d = d2.get_beta(1)
    #     d2 = d.get_beta(2)
    #     n = d2.get_node()
    #
    #     # As long as we haven't returned to the first neighbor, we keep searching.
    #     # This works because the collapse action is restricted to inner darts that are not connected to a boundary node.
    #     # Therefore, we are guaranteed to find the first vertex by following the beta1 and beta2 relations.
    #
    #     while n != n_start:
    #         adj_nodes.append(n)
    #         nodes_coord.append([n.x(), n.y()])
    #         d = d2.get_beta(1)
    #         d2 = d.get_beta(2)
    #         n = d2.get_node()
    #
    #     nodes_coord = np.array(nodes_coord)
    #
    #     # Create a Polygon with shapely package
    #     poly = Polygon(nodes_coord)
    #     # Create the point for wich we want to check star property
    #     point_v = Point(new_coordinates)
    #
    #     if plot :
    #         plt.figure(figsize=(6, 6))
    #         # Polygone
    #         x, y = poly.exterior.xy
    #         plt.fill(x, y, alpha=0.3, edgecolor='red', facecolor='lightcoral',
    #                  label='Polygon formed par by neighbours vertices')
    #
    #         # Voisins
    #         plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')
    #
    #         # Sommet testé
    #         plt.scatter(new_coordinates[0], new_coordinates[1], color='green', s=100, zorder=5, label='Vertex to test')
    #
    #         plt.legend()
    #         plt.gca().set_aspect('equal')
    #         plt.show()
    #
    #     # If polygon is convexe
    #     if poly.is_valid and poly.is_simple and poly.convex_hull.equals(poly):
    #         return True
    #     p_before = None
    #     # Si concave : vérifier visibilité
    #     for p in poly.exterior.coords[:-1]:
    #         full_seg = LineString([new_coordinates, p])
    #         new_seg_end = full_seg.interpolate(full_seg.length - 1e-5)
    #         seg = LineString([new_coordinates, new_seg_end])
    #         if not poly.contains_properly(seg):
    #             return False
    #         elif seg.crosses(poly.boundary):
    #             return False
    #         elif seg.touches(poly.boundary):
    #             return False
    #         elif p_before is not None: # test coolinearity of two vectors
    #             v1 = new_coordinates[0]-p[0], new_coordinates[1]-p[1]
    #             v2 = new_coordinates[0]-p_before[0], new_coordinates[1]-p_before[1]
    #
    #             det = v1[0] * v2[1] - v1[1] * v2[0]
    #             if -1e-5 < det < 1e-5:
    #                 return False
    #         p_before = p
    #     return Truedef is_star_vertex2(self, n1:Node, n2:Node, v):
        #plot_mesh(self.mesh)
        #
        # adj_nodes = []
        # nodes_coord = []
        # d = n1.get_dart()
        # d2 = d.get_beta(2)
        # n = d2.get_node()
        # while n != n2:
        #     adj_nodes.append(n)
        #     nodes_coord.append([n.x(), n.y()])
        #     d = d2.get_beta(1)
        #     d2 = d.get_beta(2)
        #     n = d2.get_node()
        # d = d.get_beta(1)
        # d2 = d.get_beta(2)
        # n = d2.get_node()
        # while n != n1:
        #     if n not in adj_nodes:
        #         adj_nodes.append(n)
        #         nodes_coord.append([n.x(), n.y()])
        #     d = d2.get_beta(1)
        #     d2 = d.get_beta(2)
        #     n = d2.get_node()
        # d = d.get_beta(1)
        # d2 = d.get_beta(2)
        # n = d2.get_node()
        # while n != n2:
        #     if n not in adj_nodes:
        #         adj_nodes.append(n)
        #         nodes_coord.append([n.x(), n.y()])
        #     d = d2.get_beta(1)
        #     d2 = d.get_beta(2)
        #     n = d2.get_node()
        #
        # nodes_coord = np.array(nodes_coord)
        #
        # # Créer le polygone
        # poly = Polygon(nodes_coord)
        # point_v = Point(v)
        #
        # # Vérifier si polygone est convexe
        # if poly.is_valid and poly.is_simple and poly.convex_hull.equals(poly):
        #     return True
        #
        # # Si concave : vérifier visibilité
        # for p in poly.exterior.coords[:-1]:
        #     seg = LineString([v, p])
        #     if not poly.contains(seg):
        #         # plt.figure(figsize=(6, 6))
        #         # # Polygone
        #         # x, y = poly.exterior.xy
        #         # plt.fill(x, y, alpha=0.3, edgecolor='red', facecolor='lightcoral',
        #         #          label='Polygon formed par by neighbours vertices')
        #         #
        #         # # Voisins
        #         # plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')
        #         #
        #         # # Sommet testé
        #         # plt.scatter(v[0], v[1], color='green', s=100, zorder=5, label='Vertex to test')
        #         #
        #         # plt.legend()
        #         # plt.gca().set_aspect('equal')
        #         # plt.show()
        #         return False
        # return True

    # def is_star_vertex2(self, n1:Node, n2:Node, v):
    #     #plot_mesh(self.mesh)
    #
    #     adj_nodes = []
    #     nodes_coord = []
    #     d = n1.get_dart()
    #     d2 = d.get_beta(2)
    #     n = d2.get_node()
    #     while n != n2:
    #         adj_nodes.append(n)
    #         nodes_coord.append([n.x(), n.y()])
    #         d = d2.get_beta(1)
    #         d2 = d.get_beta(2)
    #         n = d2.get_node()
    #     d = d.get_beta(1)
    #     d2 = d.get_beta(2)
    #     n = d2.get_node()
    #     while n != n1:
    #         if n not in adj_nodes:
    #             adj_nodes.append(n)
    #             nodes_coord.append([n.x(), n.y()])
    #         d = d2.get_beta(1)
    #         d2 = d.get_beta(2)
    #         n = d2.get_node()
    #     d = d.get_beta(1)
    #     d2 = d.get_beta(2)
    #     n = d2.get_node()
    #     while n != n2:
    #         if n not in adj_nodes:
    #             adj_nodes.append(n)
    #             nodes_coord.append([n.x(), n.y()])
    #         d = d2.get_beta(1)
    #         d2 = d.get_beta(2)
    #         n = d2.get_node()
    #
    #     nodes_coord = np.array(nodes_coord)
    #
    #     # Créer le polygone
    #     poly = Polygon(nodes_coord)
    #     point_v = Point(v)
    #
    #     # Vérifier si polygone est convexe
    #     if poly.is_valid and poly.is_simple and poly.convex_hull.equals(poly):
    #         return True
    #
    #     # Si concave : vérifier visibilité
    #     for p in poly.exterior.coords[:-1]:
    #         seg = LineString([v, p])
    #         if not poly.contains(seg):
    #             # plt.figure(figsize=(6, 6))
    #             # # Polygone
    #             # x, y = poly.exterior.xy
    #             # plt.fill(x, y, alpha=0.3, edgecolor='red', facecolor='lightcoral',
    #             #          label='Polygon formed par by neighbours vertices')
    #             #
    #             # # Voisins
    #             # plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Neighbours')
    #             #
    #             # # Sommet testé
    #             # plt.scatter(v[0], v[1], color='green', s=100, zorder=5, label='Vertex to test')
    #             #
    #             # plt.legend()
    #             # plt.gca().set_aspect('equal')
    #             # plt.show()
    #             return False
    #     return True

    # def find_star_vertex2(self, n1:Node, n2:Node) -> (float, float):
    #     adj_nodes = []
    #     nodes_coord = []
    #     for d_info in self.mesh.active_darts():
    #         if d_info[3] == n1.id or d_info[3] == n2.id:
    #             d2 = Dart(self.mesh, d_info[2])
    #             if d2 is not None:
    #                 n = d2.get_node()
    #                 adj_nodes.append(n)
    #                 nodes_coord.append([n.x(), n.y()])
    #             else:
    #                 raise ValueError("Collapse action may not be done near boundary")
    #     nodes_coord = np.array(nodes_coord)
    #
    #     # Ordonner les voisins autour de v
    #     vectors = nodes_coord - v
    #     angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    #     order = np.argsort(angles)
    #     neighbors_ordered = nodes_coord[order]
    #
    #     hull = ConvexHull(nodes_coord)
    #     delaunay = Delaunay(nodes_coord)
    #     plt.plot(nodes_coord[:, 0], nodes_coord[:, 1], 'o')
    #
    #     for simplex in hull.simplices:
    #         plt.plot(nodes_coord[simplex, 0], nodes_coord[simplex, 1], 'k-')
    #     plt.plot(nodes_coord[hull.vertices, 0], nodes_coord[hull.vertices, 1], 'r--', lw=2)
    #     plt.plot(nodes_coord[hull.vertices[0], 0], nodes_coord[hull.vertices[0], 1], 'ro')
    #     plt.show()
    #
    #     _ = scipy.spatial.delaunay_plot_2d(delaunay)
    #     plt.show()
    #
    #     mid = np.array([(n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2])
    #     is_star_vertex = delaunay.find_simplex(mid) >=0
    #
    #     # Calcul des angles des voisins autour de v
    #     vectors = nodes_coord - mid
    #     angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    #     order = np.argsort(angles)
    #     neighbors_ordered = nodes_coord[order]
    #
    #     # Construire le polygone
    #     poly = Polygon(neighbors_ordered)
    #
    #     # Vérifier si v est à l'intérieur ou sur la frontière
    #     point_v = Point(mid)
    #     is_star = poly.contains(point_v) or poly.touches(point_v)
    #
    #     plt.figure(figsize=(6, 6))
    #     # Polygone
    #     x, y = poly.exterior.xy
    #     plt.fill(x, y, alpha=0.3, edgecolor='red', facecolor='lightcoral', label='Polygone formé par les voisins')
    #
    #     # Voisins
    #     plt.scatter(nodes_coord[:, 0], nodes_coord[:, 1], color='blue', zorder=5, label='Voisins')
    #
    #     # Sommet testé
    #     plt.scatter(mid[0], mid[1], color='green', s=100, zorder=5, label='Sommet testé')
    #
    #     plt.legend()
    #     plt.gca().set_aspect('equal')
    #     plt.title(f"Le sommet est-il étoilé ? {is_star}")
    #     plt.show()
    #
    #     if is_star:
    #         return mid
    #     elif poly.contains(Point(n1.x(), n1.y())) or poly.touches(Point(n1.x(), n1.y())):
    #         return n1.x(), n1.y()
    #     elif poly.contains(Point(n2.x(), n2.y())) or poly.touches(Point(n2.x(), n2.y())):
    #         return n2.x(), n2.y()
    #     else:
    #         plot_mesh(self.mesh)
    #         raise ValueError("No star vertex was found")
