import math

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face
from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_analysis.global_mesh_analysis import GlobalMeshAnalysis, NodeAnalysis

FLIP_CW = 0 # flip clockwise
FLIP_CCW = 1 # flip counterclockwise
SPLIT = 2
COLLAPSE = 3
CLEANUP = 4
TEST_ALL = 5 # test if all actions are valid
ONE_VALID = 6 # test if at least one action is valid

class QuadMeshAnalysis(GlobalMeshAnalysis):
    """
    The base of quadrangular mesh analysis
    """
    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)
        #If initial scores and adjacency have not already been set
        if self.mesh.nodes[0,4] == -99:
            self.set_adjacency()
            self.set_scores()
            #self.set_geometric_quality()

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
                    ideal_adj = max(round(angle / 90) + 1, 2)
                    n.set_ideal_adjacency(ideal_adj)
                else:
                    n.set_ideal_adjacency(4)
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

class QuadMeshOldAnalysis(QuadMeshAnalysis):
    """
    Quadmesh old analysis
    """

    def __init__(self, mesh: Mesh):
        super().__init__(mesh=mesh)

    def isValidAction(self, dart_id: int, action: int) -> (bool, bool):
        """
        Test if an action is valid. You can select the ype of action between {flip clockwise, flip counterclockwise, split, collapse, cleanup, all action, one action no matter wich one}.
        :param dart_id: a dart on which to test the action
        :param action: an action type
        :return:
        """
        d = Dart(self.mesh, dart_id)
        if d.get_beta(2) is None:
            return False, True
        elif action == FLIP_CW:
            return self.isFlipCWOk(d)
        elif action == FLIP_CCW:
            return self.isFlipCCWOk(d)
        elif action == SPLIT:
            return self.isSplitOk(d)
        elif action == COLLAPSE:
            return self.isCollapseOk(d)
        elif action == CLEANUP:
            return self.isCleanupOk(d)
        elif action == TEST_ALL:
            topo, geo = self.isFlipCCWOk(d)
            if not (topo and geo):
                return False, False
            topo, geo = self.isFlipCWOk(d)
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
            topo_flip, geo_flip = self.isFlipCCWOk(d)
            if (topo_flip and geo_flip):
                return True, True
            topo_flip, geo_flip = self.isFlipCWOk(d)
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


    def isFlipCCWOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True

        # if d is on boundary, flip is not possible
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n5_analysis = NodeAnalysis(n5)
        n3_analysis = NodeAnalysis(n3)
        # if degree will not too high
        if not n5_analysis.test_degree() or not n3_analysis.test_degree():
            topo = False
            return topo, geo

        # if two faces share two edges
        if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
            topo = False
            return topo, geo

        # check validity of the two modified quads
        geo = self.isValidQuad(n5, n6, n2, n3) and self.isValidQuad(n1, n5, n3, n4)

        return topo, geo

    def isFlipCWOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True
        # if d is on boundary, flip is not possible
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n4_analysis = NodeAnalysis(n4)
        n6_analysis = NodeAnalysis(n6)
        if not n4_analysis.test_degree() or not n6_analysis.test_degree():
            topo = False
            return topo, geo

        if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
            topo = False
            return topo, geo
        geo = self.isValidQuad(n4, n6, n2, n3) and self.isValidQuad(n1, n5, n6, n4)

        return topo, geo


    def isSplitOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n4_analysis = NodeAnalysis(n4)
        n2_analysis = NodeAnalysis(n2)
        if not n4_analysis.test_degree() or not n2_analysis.test_degree():
            topo = False
            return topo, geo

        if d211.get_node() == d111.get_node() or d11.get_node() == d2111.get_node():
            topo = False
            return topo, geo

        n10 = self.mesh.add_node((n1.x() + n2.x()) / 2, (n1.y() + n2.y()) / 2)
        geo = self.isValidQuad(n4, n1, n5, n10) and self.isValidQuad(n4, n10, n2, n3) and self.isValidQuad(n10, n5, n6, n2)
        self.mesh.del_node(n10)
        return topo, geo


    def isCollapseOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True
        if d.get_beta(2) is None:
            topo = False
            return topo, geo
        else:
            d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6 = self.mesh.active_quadrangles(d)

        n1_analysis = NodeAnalysis(n1)
        n3_analysis = NodeAnalysis(n3)

        if n1_analysis.on_boundary():
            topo = False
            return topo, geo

        if (n3_analysis.degree() +n1_analysis.degree()-2) > 10:
            topo = False
            return topo, geo

        adjacent_faces_lst = []
        f1 = d2.get_face()
        adjacent_faces_lst.append(f1.id)
        d12 = d1.get_beta(2)
        if d12 is not None:
            f2 = d12.get_face()
            adjacent_faces_lst.append(f2.id)
        d112 = d11.get_beta(2)
        if d112 is not None:
            f3 = d112.get_face()
            adjacent_faces_lst.append(f3.id)
        d1112 = d111.get_beta(2)
        if d1112 is not None:
            f4 = d1112.get_face()
            adjacent_faces_lst.append(f4.id)

        # Check that there are no adjacent faces in common
        if len(adjacent_faces_lst) != len(set(adjacent_faces_lst)):
            topo = False
            return topo, geo

        adj_faces = n3_analysis.adjacent_faces_id()
        adj_faces.extend(n1_analysis.adjacent_faces_id())

        #If the opposite vertex is on the edge, it is not moved

        if n3_analysis.on_boundary():
            n10 = self.mesh.add_node( n3.x(), n3.y())
        else:
            n10 = self.mesh.add_node((n1.x() + n3.x()) / 2, (n1.y() + n3.y()) / 2)

        for f_id in adj_faces:
            if f_id != (d.get_face()).id:
                f = Face(self.mesh, f_id)
                df = f.get_dart()
                df1 = df.get_beta(1)
                df11 = df1.get_beta(1)
                df111 = df11.get_beta(1)
                A = df.get_node()
                B = df1.get_node()
                C = df11.get_node()
                D = df111.get_node()
                if A==n1 or A==n3:
                    A=n10
                elif B==n1 or B==n3:
                    B=n10
                elif C==n1 or C==n3:
                    C=n10
                elif D==n1 or D==n3:
                    D=n10

                if not self.isValidQuad(A, B, C, D):
                    geo = False
                    self.mesh.del_node(n10)
                    return topo, geo

        self.mesh.del_node(n10)
        return topo, geo


    def isCleanupOk(self, d: Dart) -> (bool, bool):
        topo = True
        geo = True
        if d.get_beta(2) is None:
            topo = False
        mesh = d.mesh
        parallel_darts = mesh.find_parallel_darts(d)
        for d in parallel_darts:
            d111 = ((d.get_beta(1)).get_beta(1)).get_beta(1)
            if d111.get_beta(2) is None:
                topo = False
                return topo, geo
        return topo, geo


    def isTruncated(self, darts_list)-> bool:
        for d_id in darts_list:
            if self.isValidAction(d_id, 4)[0]:
                return False
        return True

    def isValidQuad(self, A: Node, B: Node, C: Node, D: Node):
        u1 = np.array([B.x() - A.x(), B.y() - A.y()]) # vect(AB)
        u2 = np.array([C.x() - B.x(), C.y() - B.y()]) # vect(BC)
        u3 = np.array([D.x() - C.x(), D.y() - C.y()]) # vect(CD)
        u4 = np.array([A.x() - D.x(), A.y() - D.y()]) # vect(DA)

        # Checking for near-zero vectors (close to (0,0))
        if (np.linalg.norm(u1) < 1e-5 or
                np.linalg.norm(u2) < 1e-5 or
                np.linalg.norm(u3) < 1e-5 or
                np.linalg.norm(u4) < 1e-5):
            return False  # Quad invalid because one side is almost zero

        cp_A = self.cross_product(-1*u4, u1)
        cp_B = self.cross_product(-1*u1, u2)
        cp_C = self.cross_product(-1*u2, u3)
        cp_D = self.cross_product(-1*u3, u4)

        zero_count = sum(-1e-5<cp<1e-5 for cp in [cp_A, cp_B, cp_C, cp_D])
        if zero_count>=2:
            return False
        elif 0<= self.signe(cp_A)+ self.signe(cp_B)+ self.signe(cp_C)+ self.signe(cp_D) <2 :
            return True
        else:
            return False

    def signe(self, a: int):
        if a <= 0: # Before it was 1e-8
            return 0
        else:
            return 1