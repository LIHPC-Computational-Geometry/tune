from __future__ import annotations
import sys
import numpy

from mesh_model.mesh_struct.mesh_elements import Dart, Node, Face

"""
Classes Dart, Node and Face must be seen as handlers on data that are stored in the
Mesh class. 
"""


class Mesh:
    def __init__(self, nodes=[], faces=[]):
        """
        Vertices are stored in a numpy array containing coordinates (x,y, dart id, ideal adjacency, vertex score)
        Faces are stored in a numpy array of simple (dart ids)
        Darts are stored in a numpy array, where each dart is a 5-tuple (dart id, beta_1, beta_2, vertex_id, face_id, geo_quality)
        Ideal adjacency, vertex scores and geometric quality are not defined here. You must use Mesh analysis class to define them
        """
        self.nodes = numpy.empty((0, 5), dtype=float)
        self.faces = numpy.empty(0, dtype=int)
        self.dart_info = numpy.empty((0, 6), dtype=int)
        self.first_free_dart = 0
        self.first_free_node = 0
        self.first_free_face = 0

        for n in nodes:
            self.add_node(n[0], n[1])

        for f in faces:
            if len(f) == 3:
                self.add_triangle(Node(self, f[0]),
                                  Node(self, f[1]),
                                  Node(self, f[2]))

            elif len(f) == 4:
                self.add_quad(Node(self, f[0]),
                              Node(self, f[1]),
                              Node(self, f[2]),
                              Node(self, f[3]))
            else:
                raise ValueError("Only triangles and quads are supported")

        # now we 2-sew some darts to glue faces along edges
        self.set_twin_pointers()

    def nb_nodes(self) -> int:
        """
        :return: the number of vertices in the mesh
        """
        # We filter the vertices having the x-coordinate equals to max float. Such vertices were removed
        return len(self.active_nodes())

    def nb_faces(self) -> int:
        """
           :return: the number of faces in the mesh
           """
        # We filter the faces having the -1 value. An item with this value is a deleted face
        return len(self.active_faces())

    def add_node(self, x: float, y: float) -> Node:
        """
        Add a vertex in the mesh, this node is not connected to a dart here
        The ideal adjacency and vertex score are not defined here
        :param x: X coordinate
        :param y: Y coordinate
        :return: the created node
        """
        if len(self.nodes) <= self.first_free_node:
            self.nodes = numpy.append(self.nodes, [[x, y, -1, -1, -99]], axis=0)
            self.first_free_node += 1
            return Node(self, len(self.nodes) - 1)
        elif self.first_free_node >= 0:
            n_id = int(self.first_free_node)
            if isinstance(n_id, int):
                self.first_free_node = abs(self.nodes[n_id, 2] + 1)
                self.nodes[n_id] = [x, y, -1, -1, -99]
            else:
                print(n_id)
                print(type(n_id))
                raise ValueError("n_id not integer")
            return Node(self, n_id)
        else:
            raise ValueError("Try to  add a node outside the array")

    def del_node(self, n: Node) -> None:
        self.nodes[n.id, 2] = -self.first_free_node - 1
        self.first_free_node = n.id

    def del_vertex(self, ni: Node) -> None:
        """
        Removes the node ni. Warning all the darts that point
        to this node will be invalid (but not automatically updated)
        :param ni: a node
        """
        ni.set_x(sys.float_info.max)
        self.del_node(ni)

    def add_triangle(self, n1: Node, n2: Node, n3: Node) -> Face:
        """
        Add a triangle defined by nodes of indices n1, n2, and n3.
        The triangle is created in the order of n1, n2 then n3. Internally,
        the created triangle points to the dart that goes from n1 to n2.
        An exception is raised if one of the nodes does not exist
        :param n1: first node
        :param n2: second node
        :param n3: third node
        :return: the id of the triangle
        """
        nodes = [n1, n2, n3]
        # create 3 darts
        darts = [self.add_dart() for i in range(3)]

        darts[0].set_beta(1, darts[1])
        darts[1].set_beta(1, darts[2])
        darts[2].set_beta(1, darts[0])

        for k in range(3):
            darts[k].set_node(nodes[k])
            nodes[k].set_dart(darts[k])

        if len(self.faces) <= self.first_free_face:
            self.faces = numpy.append(self.faces, [darts[0].id])
            self.first_free_face += 1
            tri = Face(self, len(self.faces) - 1)
        elif self.first_free_face >= 0:
            f_id = self.first_free_face
            self.first_free_face = abs(self.faces[f_id]+1)
            self.faces[f_id] = darts[0].id
            tri = Face(self, f_id)
        else:
            raise ValueError("Try to  add a node outside the array")

        for d in darts:
            d.set_face(tri)

        return tri

    def del_triangle(self, d1: Dart, d2: Dart, d3: Dart, f: Face) -> None:
        self.del_dart(d1)
        self.del_dart(d2)
        self.del_dart(d3)

        self.faces[f.id] = -self.first_free_face - 1
        self.first_free_face = f.id

    def del_adj_triangles(self, d: Dart) -> None:
        """
        Delete the two adjacent triangles of the given dart d
        :param mesh: a mesh
        :param d: the dart to be deleted
        """
        d2 = d.get_beta(2)
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d21 = d2.get_beta(1)
        d211 = d21.get_beta(1)

        f1 = d.get_face()
        f2 = d2.get_face()

        self.del_triangle(d, d1, d11, f1)
        self.del_triangle(d2, d21, d211, f2)

    def add_quad(self, n1: Node, n2: Node, n3: Node, n4: Node) -> Face:
        """
        Add a quad defined by nodes of indices n1, n2, n3 and n4.
        The quad is created in the order of n1, n2, n3 then n4. Internally,
        the created quad points to the dart that goes from n1 to n2.
        An exception is raised if one of the nodes does not exist
        :param n1: first node
        :param n2: second node
        :param n3: third node
        :param n4: fourth node
        :return: the id of the quad
        """
        nodes = [n1, n2, n3, n4]
        if inverseQuad(n1, n2, n3, n4):
            nodes = [n4, n3, n2, n1]
        elif not isValidQuad(n1, n2, n3, n4):
            raise ValueError("Quad not valid")

        # create 4 darts
        darts = [self.add_dart() for i in range(4)]

        darts[0].set_beta(1, darts[1])
        darts[1].set_beta(1, darts[2])
        darts[2].set_beta(1, darts[3])
        darts[3].set_beta(1, darts[0])

        for k in range(4):
            darts[k].set_node(nodes[k])
            nodes[k].set_dart(darts[k])

        self.faces = numpy.append(self.faces, [darts[0].id])
        quad = Face(self, len(self.faces) - 1)

        for d in darts:
            d.set_face(quad)

        return quad

    def del_quad(self, d1: Dart, d2: Dart, d3: Dart, d4: Dart, f: Face) -> None:
        self.del_dart(d1)
        self.del_dart(d2)
        self.del_dart(d3)
        self.del_dart(d4)

        self.faces[f.id] = -self.first_free_face - 1
        self.first_free_face = f.id

    def set_twin_pointers(self) -> None:
        """
        This function search for the inner darts to connect and connect them with beta2.
        """
        for d_info in self.active_darts():
            d = Dart(self, d_info[0])
            if d.get_beta(2) is None:
                # d is not 2-sew, we look for a dart to connect. If we don'f find one,
                # it means we are on the mesh_struct boundary

                d_nfrom = d.get_node()
                d_nto = d.get_beta(1).get_node()

                for d2_info in self.active_darts():
                    d2 = Dart(self, d2_info[0])
                    if d2.get_beta(2) is None:
                        d2_nfrom = d2.get_node()
                        d2_nto = d2.get_beta(1).get_node()
                        if d_nfrom.id == d2_nto.id and d_nto.id == d2_nfrom.id:
                            d.set_beta(2, d2)
                            d2.set_beta(2, d)

    def find_inner_edge(self, n1: Node, n2: Node) -> (bool, Dart):
        """
        Try and find the edge connecting n1 and n2. If the edge does
        not exist, or is on the mesh boundary, it returns False. Else
        it returns True and the dart coming from n1 and pointing to n2
        :param n1: First node
        :param n2: Second node
        :return: the inner dart connecting n1 to n2 if it exists
        """
        for d_info in self.active_darts():
            d = Dart(self, d_info[0])
            d2 = d.get_beta(2)
            if d2 is not None:
                if d.get_node().id == n1.id and d2.get_node().id == n2.id:
                    return True, d
                if d.get_node().id == n2.id and d2.get_node().id == n1.id:
                    return True, d2
        return False, None

    def set_face_beta2(self, f: Face, darts: list[Dart]) -> None:
        """
        Set beta2 relation between darts and face darts when possible
        :param f: the face
        :param darts: to darts to try to connect by beta2
        """
        df_current = f.get_dart()
        end = False
        while not end:
            nf_start = df_current.get_node()
            nf_end = df_current.get_beta(1).get_node()

            for d in darts:
                if d is not None:
                    nd_start = d.get_node()
                    nd_end = d.get_beta(1).get_node()
                    if nf_start == nd_end and nf_end == nd_start:
                        d.set_beta(2, df_current)
                        df_current.set_beta(2, d)

            df_current = df_current.get_beta(1)
            end = (df_current.id == f.get_dart().id)

    def add_dart(self, a1: int = -1, a2: int = -1, v: int = -1, f: int = -1, q: int = -99) -> Dart:
        """
        This function add a dart in the mesh. It must not be used directly
        :param a1: dart index to connect by alpha1
        :param a2: dart index to connect by alpha2
        :param v:  vertex index this dart point to
        :param f: face to connect
        :param q: geometric quality around the dart
        :return: the created dart
        """
        if len(self.dart_info) <= self.first_free_dart:
            self.dart_info = numpy.append(self.dart_info, [[len(self.dart_info), a1, a2, v, f, q]], axis=0)
            self.first_free_dart += 1
            return Dart(self, len(self.dart_info) - 1)
        elif len(self.dart_info) > self.first_free_dart:
            next_free_dart = abs(self.dart_info[self.first_free_dart][0]+1)
            dart_id = self.first_free_dart
            self.dart_info[dart_id][0] = dart_id
            self.dart_info[dart_id][1] = a1
            self.dart_info[dart_id][2] = a2
            self.dart_info[dart_id][3] = v
            self.dart_info[dart_id][4] = f
            self.first_free_dart = next_free_dart
            return Dart(self, dart_id)
        else:
            raise IndexError('Dart index out of range')

    def del_dart(self, d: Dart):
        """
        n = d.get_node()
        if n.get_dart() == d:
            d2 = d.get_beta(2)
            d21 = d2.get_beta(1)
            n.set_dart(d21)
        """
        self.dart_info[d.id][0] = -self.first_free_dart - 1
        self.first_free_dart = d.id
        d.active = False

    def is_dart_active(self, d: Dart) -> bool:
        """Check if the dart has been deleted"""
        if d is None:
            return False
        elif self.dart_info[d.id,0] < 0:
            return False
        elif self.dart_info[d.id,0] >=0 :
            return True

    def set_beta2(self, dart: Dart) -> None:
        """
        Search for a dart to connect with beta2 relation when possible.
        :param dart: the dart to connect with a beta2 relation
        """
        dart_nfrom = dart.get_node()
        dart_nto = dart.get_beta(1)
        for d_info in self.active_darts():
            d = Dart(dart.mesh, d_info[0])
            d_nfrom = d.get_node()
            d_nto = d.get_beta(1)
            if d_nfrom == dart_nto.get_node() and d_nto.get_node() == dart_nfrom :
                d.set_beta(2, dart)
                dart.set_beta(2, d)

    def active_nodes(self):
        return self.nodes[self.nodes[:, 2] >= 0]

    def active_darts(self):
        return self.dart_info[self.dart_info[:, 0] >= 0]

    def active_faces(self):
        return self.faces[self.faces[:] >= 0]

    def active_triangles(self, d: Dart) -> tuple[Dart, Dart, Dart, Dart, Dart, Node, Node, Node, Node]:
        """
        Return the darts and nodes around selected dart
        :param mesh: the mesh
        :param d: selected dart
        :return: a tuple of darts and nodes
        """
        d2 = d.get_beta(2)
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d21 = d2.get_beta(1)
        d211 = d21.get_beta(1)
        n1 = d.get_node()
        n2 = d2.get_node()
        n3 = d11.get_node()
        n4 = d211.get_node()

        return d2, d1, d11, d21, d211, n1, n2, n3, n4

    def active_quadrangles(self, d: Dart) -> tuple[Dart, Dart, Dart, Dart, Dart, Dart, Dart, Node, Node, Node, Node, Node, Node]:
        """
        Return the darts and nodes around selected dart
        :param mesh: the mesh
        :param d: selected dart
        :return: a tuple of darts and nodes
        """
        d2 = d.get_beta(2)
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d111 = d11.get_beta(1)
        d21 = d2.get_beta(1)
        d211 = d21.get_beta(1)
        d2111 = d211.get_beta(1)
        n1 = d.get_node()
        n2 = d2.get_node()
        n3 = d11.get_node()
        n4 = d111.get_node()
        n5 = d211.get_node()
        n6 = d2111.get_node()

        return d2, d1, d11, d111, d21, d211, d2111, n1, n2, n3, n4, n5, n6

    def find_parallel_darts(self, d: Dart) -> list[Dart]:
        parallel_darts = []
        dp = d
        #sens 1
        while dp is not None:
            dp2 = dp.get_beta(2)
            if dp2 is None:
                dp = None
            else:
                dp21 = dp2.get_beta(1)
                dp = dp21.get_beta(1)
                if dp not in parallel_darts :
                    parallel_darts.append(dp)
                else:
                    dp = None

        #sens 2
        dp = d
        while dp is not None:
            if dp not in parallel_darts :
                parallel_darts.append(dp)
                dp1 = dp.get_beta(1)
                dp11 = dp1.get_beta(1)
                dp = dp11.get_beta(2)
            else :
                dp = None

        return parallel_darts



def inverseQuad(A: Node, B: Node, C: Node, D: Node):
    u1 = numpy.array([B.x() - A.x(), B.y() - A.y()]) # vect(AB)
    u2 = numpy.array([C.x() - B.x(), C.y() - B.y()]) # vect(BC)
    u3 = numpy.array([D.x() - C.x(), D.y() - C.y()]) # vect(CD)
    u4 = numpy.array([A.x() - D.x(), A.y() - D.y()]) # vect(DA)

    cp_A = cross_product(-1*u4, u1)
    cp_B = cross_product(-1*u1, u2)
    cp_C = cross_product(-1*u2, u3)
    cp_D = cross_product(-1*u3, u4)

    if cp_A >=0 and cp_B>=0 and cp_C>=0 and cp_D>=0:
        return True
    else:
        return False

def cross_product(vect_AB, vect_AC):
    """ Return the cross product between AB et AC.
        0 means A, B and C are coolinear
        > 0 mean A, B and C are "sens des aiguilles d'une montre"
        < 0 sens inverse
    """
    val = vect_AB[0] * vect_AC[1] - vect_AB[1] * vect_AC[0]
    return val

def signe(a: int):
    if a<=0:
        return 0
    else:
        return 1

def isValidQuad(A: Node, B: Node, C: Node, D: Node):
    u1 = numpy.array([B.x() - A.x(), B.y() - A.y()]) # vect(AB)
    u2 = numpy.array([C.x() - B.x(), C.y() - B.y()]) # vect(BC)
    u3 = numpy.array([D.x() - C.x(), D.y() - C.y()]) # vect(CD)
    u4 = numpy.array([A.x() - D.x(), A.y() - D.y()]) # vect(DA)

    cp_A = cross_product(-1*u4, u1)
    cp_B = cross_product(-1*u1, u2)
    cp_C = cross_product(-1*u2, u3)
    cp_D = cross_product(-1*u3, u4)

    if signe(cp_A)+signe(cp_B)+signe(cp_C)+signe(cp_D)<2:
        return True
    else:
        return False