from math import radians, cos, sin

import numpy as np
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
            * quality = 2: crossed quad
            * quality = 3: flat or "ecrase" quad
            * quality = 4: one adjacent triangular face is flat
            * quality = 5: triangular quad
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
        if (np.linalg.norm(u2) < 1e-5 or
                np.linalg.norm(u3) < 1e-5 or
                np.linalg.norm(u4) < 1e-5 or
                np.linalg.norm(u5) < 1e-5):
            plot_mesh(self.mesh)
            #raise ValueError("near zero vector") # Quad invalid because one side is almost zero

        # Calculate cross product at each node
        cp_A = self.cross_product(-1 * u3, u4)
        cp_D = self.cross_product(-1 * u4, u5)
        cp_B = self.cross_product(-1 * u5, u2)
        cp_C = self.cross_product(-1 * u2, u3)

        # Counts how many angles are oriented clockwise. If the angle is clockwise oriented, signe(cp) return 1
        sum_cp = self.signe(cp_A) + self.signe(cp_B) + self.signe(cp_C) + self.signe(cp_D)


        zero_count = sum(-1e-5 < cp < 1e-5 for cp in [cp_A, cp_B, cp_C, cp_D])
        if zero_count >= 2: #If two cross product are near zero, it means it's a flat quadrangle and the two adjacent faces are flat
            plot_mesh(self.mesh)
            return 3 #flat
        elif zero_count == 1 and sum_cp == 1:
            plot_mesh(m)
            plot_mesh(self.mesh)
            return 6
            #plot_mesh(self.mesh)
            #print("1")
        elif zero_count == 1 and sum_cp == 2:
            plot_mesh(self.mesh)
            return 4 #half flat one face is flat and the other is triangular
        elif zero_count == 1 and sum_cp == 0:
            #plot_mesh(self.mesh)
            return 5 # The two adjacent triangular faces form a triangular quad face

        if sum_cp == 0:
            return 0 #convexe
        elif sum_cp == 1:
            return 1 #concave
        elif sum_cp == 2:
            return 2 #crossed
        elif sum_cp > 2:
            plot_mesh(self.mesh)
        if m is not None:
            plot_mesh(m)
        plot_mesh(self.mesh)

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

        if d.get_quality() in [2,3,4]: # if the face around is crossed or flat, or half flat
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


        if d.get_quality() in [2, 3, 4]:  # if the face around is crossed or flat, or half flat
            geo = False
        elif d1.get_quality() == 1 or d11.get_quality() == 1 or d21.get_quality() == 1 or d211.get_quality() == 1:
            geo = False

        return topo, geo

    def isTruncated(self, darts_list) -> bool:
        for d_id in darts_list:
            if self.isValidAction( d_id, 4)[0]:
                return False
        return True