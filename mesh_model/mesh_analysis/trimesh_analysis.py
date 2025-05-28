from math import sqrt, degrees, radians, cos, sin
from mesh_model.mesh_analysis.global_mesh_analysis import GlobalMeshAnalysis, NodeAnalysis
from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh


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

    def isValidAction(self, dart_id: int, action: int) -> (bool,bool):
        pass

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


class TriMeshTopoAnalysis(TriMeshAnalysis):
    """
    The base of triangular mesh analysis
    """
    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)

    def isValidAction(self, dart_id: int, action: int) -> (bool,bool):
        d = Dart(self.mesh, dart_id)
        boundary_darts = self.get_boundary_darts()

        geo = True # The geometric validity is automatically set to True, it is not tested here

        if d in boundary_darts:
            return False, geo
        elif action == FLIP:
            return self.isFlipOk(d), geo
        elif action == SPLIT:
            return self.isSplitOk(d), geo
        elif action == COLLAPSE:
            return self.isCollapseOk(d), geo
        elif action == TEST_ALL:
            return (not self.isFlipOk(d) and not self.isSplitOk(d) and not self.isCollapseOk(d)), geo
        elif action == ONE_VALID:
            return (self.isFlipOk(d) or self.isSplitOk(d) or self.isCollapseOk(d)), geo
        else:
            raise ValueError("No valid action")

    def isFlipOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True

        #if d is on boundary, flip is not possible
        if d.get_beta(2) is None:
            topo = False
            return topo, geo

        _, _, _, _, _, A, B, C, D = self.mesh.active_triangles(d)

        nA_analysis = NodeAnalysis(A)
        nB_analysis = NodeAnalysis(B)

        if not nA_analysis.test_degree() or not nB_analysis.test_degree():
            topo = False

        #Edge reversal allowed

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
        if d112 is None or d12 is None or d2112 is None or d212 is None:
            topo = False
        elif n1_analysis.on_boundary() or n2_analysis.on_boundary():
            topo = False
        elif not n1_analysis.test_degree():
            topo = False

        return topo, geo

    def isTruncated(self, darts_list) -> bool:
        for d_id in darts_list:
            if self.isValidAction( d_id, 4)[0]:
                return False
        return True


class TriMeshGeoAnalysis(TriMeshAnalysis):
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
        dist_AB = sqrt(vect_AB[0] ** 2 + vect_AB[1] ** 2)
        dist_AC = sqrt(vect_AC[0] ** 2 + vect_AC[1] ** 2)
        dist_BC = sqrt(vect_BC[0] ** 2 + vect_BC[1] ** 2)
        target_mesh_size = 1

        L_max = max(dist_AB, dist_AC, dist_BC)

        if target_mesh_size / 2 * sqrt(2) < L_max and L_max < target_mesh_size * 3 * sqrt(2):  # 0.35<Lmax<4.24
            pass
        else:
            return False

        # Calcul des angles avec le théorème du cosinus
        angle_B = degrees(self.angle_from_sides(dist_AC, dist_AB, dist_BC))  # Angle au point A
        angle_C = degrees(self.angle_from_sides(dist_AB, dist_BC, dist_AC))  # Angle au point B
        angle_A = degrees(self.angle_from_sides(dist_BC, dist_AC, dist_AB))  # Angle au point C

        # Vérification que tous les angles sont supérieurs à 5°
        if angle_A <= 5 or angle_B <= 5 or angle_C <= 5:
            return False
        return True

    def isTruncated(self, darts_list) -> bool:
        for d_id in darts_list:
            if self.isValidAction(d_id, 4)[0]: # if on action is valid, it means it's valid topologically and geometrically, so no need to verify the two
                return False
        return True