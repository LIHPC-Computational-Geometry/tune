from __future__ import annotations
import sys
import numpy

from model.mesh_struct.mesh_elements import Dart, Node, Face

"""
Classes Dart, Node and Face must be seen as handlers on data that are stored in the
Mesh class. 
"""


class Mesh:
    def __init__(self, nodes=[], faces=[]):
        """
        Vertices are stored in a numpy array containing coordinates (x,y, dart id)
        Faces are stored in a numpy array of simple (dart ids)
        Darts are stored in a numpy array, where each dart is a 5-tuple (dart id, beta_1, beta_2, vertex_id, face_id)
        """
        self.nodes = numpy.empty((0, 3))
        self.faces = numpy.empty(0, dtype=int)
        self.dart_info = numpy.empty((0, 5), dtype=int)

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
        return len(self.nodes[self.nodes[:, 0] != sys.float_info.max])

    def nb_faces(self) -> int:
        """
           :return: the number of faces in the mesh
           """
        # We filter the faces having the -1 value. An item with this value is a deleted face
        return len(self.faces[self.faces[:] != -1])

    def add_node(self, x: float, y: float) -> Node:
        """
        Add a vertex in the mesh, this node is not connected to a dart here
        :param x: X coordinate
        :param y: Y coordinate
        :return: the created node
        """
        self.nodes = numpy.append(self.nodes, [[x, y, -1]], axis=0)
        return Node(self, len(self.nodes) - 1)

    def del_vertex(self, ni: int) -> None:
        """
        Removes the node ni. Warning all the darts that point
        to this node will be invalid (but not automatically updated)
        :param ni: a node
        """
        ni.set_x(sys.float_info.max)

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

        self.faces = numpy.append(self.faces, [darts[0].id])
        tri = Face(self, len(self.faces)-1)

        for d in darts:
            d.set_face(tri)

        return tri

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

    def set_twin_pointers(self) -> None:
        """
        This function search for the inner darts to connect and connect them with beta2.
        """
        for d_info in self.dart_info:
            d = Dart(self, d_info[0])
            if d.get_beta(2) is None:
                # d is not 2-sew, we look for a dart to connect. If we don'f find one,
                # it means we are on the mesh_struct boundary

                d_nfrom = d.get_node()
                d_nto = d.get_beta(1).get_node()

                for d2_info in self.dart_info:
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
        for d_info in self.dart_info:
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
                nd_start = d.get_node()
                nd_end = d.get_beta(1).get_node()
                if nf_start == nd_end and nf_end == nd_start:
                    d.set_beta(2, df_current)
                    df_current.set_beta(2, d)

            df_current = df_current.get_beta(1)
            end = (df_current.id == f.get_dart().id)

    def add_dart(self, a1: int = -1, a2: int = -1, v: int = -1, f: int = -1) -> Dart:
        """
        This function add a dart in the mesh. It must not be used directly
        :param a1: dart index to connect by alpha1
        :param a2: dart index to connect by alpha2
        :param v:  vertex index this dart point to
        :return: the created dart
        """
        self.dart_info = numpy.append(self.dart_info, [[len(self.dart_info), a1, a2, v, f]], axis=0)
        return Dart(self, len(self.dart_info) - 1)

    def set_beta2(self, dart: Dart) -> None:
        """
        Search for a dart to connect with beta2 relation when possible.
        :param dart: the dart to connect with a beta2 relation
        """
        dart_nfrom = dart.get_node()
        dart_nto = dart.get_beta(1)
        for d_info in dart.mesh.dart_info:
            d = Dart(dart.mesh, d_info[0])
            d_nfrom = d.get_node()
            d_nto = d.get_beta(1)
            if d_nfrom == dart_nto.get_node() and d_nto.get_node() == dart_nfrom :
                d.set_beta(2, dart)
                dart.set_beta(2, d)

