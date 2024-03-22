from __future__ import annotations
import sys
import numpy

"""
Classes Dart, Node and Face must be seen as handlers on data that are stored in the
mesh class. 
"""


class Dart:
    def __init__(self, m, id: int):
        """
        A dart is defined from the mesh it belongs to, and an id, which is the index
        in the mesh container where the dart is stored. In this container, the dart d may be
        defined by 4 fields:
        - the id of the beta1(d),
        - the id of beta2(d)
        - the id of the node d is coming from
        - the id of the face d it belongs to
        When a dart is created, it doesn't know any node or face
        :param m: a mesh
        :param id: a index, that corresponds to the location of the dart data in the mesh dart container
        """
        self.mesh = m
        self.id = id

    def get_beta(self, i: int) -> Dart:
        """
        Get the dart connected to by alpha_i
        :param i: alpha dimension
        :return: the id of the dart alpha_i(self)
        """
        if i < 1 or i > 2:
            raise ValueError("Wrong alpha dimension")
        return Dart(self.mesh, self.mesh.darts[self.id - 1, i - 1])

    def set_beta(self, i: int, dart_to: Dart) -> None:
        if i < 1 or i > 2:
            raise ValueError("Wrong alpha dimension")
        self.mesh.darts[self.id - 1, i - 1] = dart_to.id

    def get_node(self) -> Node:
        node_id = self.mesh.darts[self.id - 1, 2]
        if node_id == -1:
            raise ValueError("No associated node found")
        return Node(self.mesh, node_id)

    def set_node(self, node: Node) -> None:
        self.mesh.darts[self.id - 1, 2] = node.id

    def get_face(self) -> Face:
        face_id = self.mesh.darts[self.id - 1, 3]
        if face_id == -1:
            raise ValueError("No associated face found")
        return Face(self.mesh, face_id)

    def set_face(self, face: Face) -> None:
        self.mesh.darts[self.id - 1, 3] = face.id


class Node:
    def __init__(self, m: Mesh, id: int):
        self.mesh = m
        self.id = id

    def __eq__(self, n):
        return self.mesh == n.mesh and self.id == n.id

    def set_dart(self, dart: Dart) -> None:
        """
        Update the dart value associated with this node
        :param dart_index: the index of the dart in self.mesh
        """
        self.mesh.nodes[self.id - 1, 2] = dart.id

    def get_dart(self) -> Dart:
        return Dart(self.mesh, self.mesh.nodes[self.id - 1, 2])

    def x(self) -> float:
        return self.mesh.nodes[self.id - 1, 0]

    def y(self) -> float:
        return self.mesh.nodes[self.id - 1, 1]

    def xy(self) -> float:
        return self.mesh.nodes[self.id - 1]

    def set_x(self, x: float) -> None:
        self.mesh.nodes[self.id - 1, 0] = x

    def set_y(self, y: float) -> None:
        self.mesh.nodes[self.id - 1, 1] = y

    def set_xy(self, x: float, y: float) -> None:
        self.set_x(x)
        self.set_y(y)


class Face:
    def __init__(self, m: Mesh, id: int):
        """
        :param m: the mesh that contains the current face
        :param id: the face id
        We do not provide id for each face. A face is "fully" defined by its owning dart.
        The value self.mesh.faces[self.id - 1] is the dart corresponding to self.
        """
        self.mesh = m
        self.id = id

    def get_nodes(self) -> list[Node]:
        start = self.get_dart()
        l = []
        while start.get_beta(1) != self.mesh.faces[self.id - 1]:
            l.append(start.get_node())
            start = Dart(self.mesh, start.get_beta(1))
        return l

    def get_dart(self) -> Dart:
        return Dart(self.mesh, self.mesh.faces[self.id - 1])

    def set_dart(self, dart: Dart) -> None:
        self.mesh.faces[self.id - 1] = dart.id


class Mesh:
    def __init__(self):
        """
        Vertices are stored in a numpy array containing coordinates (x,y, dart id)
        Faces are stored in a numpy array of simple (dart ids)
        Darts are stored in a numpy array, where each dart is a 4-tuple (beta_1, beta_2, vertex_id, face_id)
        """
        self.nodes = numpy.empty((0, 3))
        self.faces = numpy.empty(0, dtype=int)
        self.darts = numpy.empty((0, 4), dtype=int)

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
        self.nodes = numpy.append(self.nodes, [[x, y,-1]], axis=0)
        return Node(self, len(self.nodes))

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
        the created triangle points on the dart that goes from n1 to n2.
        An exception is raised if one of the nodes does not exist
        :param n1: first node
        :param n2: second node
        :param n3: third node
        :return: the id of the triangle
        """
        # create darts
        d1 = self.add_dart()
        d2 = self.add_dart()
        d3 = self.add_dart()

        d1.set_beta(1, d2)
        d2.set_beta(1, d3)
        d3.set_beta(1, d1)

        d1.set_node(n1)
        d2.set_node(n2)
        d3.set_node(n3)

        n1.set_dart(d1)
        n2.set_dart(d2)
        n3.set_dart(d3)

        self.faces = numpy.append(self.faces, [d1.id])
        tri = Face(self, len(self.faces))

        d1.set_face(tri)
        d2.set_face(tri)
        d3.set_face(tri)

        return tri

    def find_inner_edge(self, n1: Node, n2: Node) -> (bool, Dart):
        """
        Try and find the edge connecting n1 and n2. If the edge does
        not exist, or is on the mesh boundary, it returns False. Else
        it returns True and the dart coming from n1 and pointing to n2
        :param n1: First node
        :param n2: Second node
        :return: the inner dart connecting n1 to n2 if it exists
        """
        for d_info in self.darts:
            d = Dart(self, d_info[0])
            d2 = d.get_beta(2)
            if d2.id != -1:
                if d.get_node().id == n1.id and d2.get_node().id == n2.id:
                    return True, d
                if d.get_node().id == n2.id and d2.get_node().id == n1.id:
                    return True, d2
        return False, None

    def swap_edge(self, n1: Node, n2: Node) -> bool:
        found, d = self.find_inner_edge(n1, n2)
        if not found:
            return False
        d2 = d.get_beta(2)
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d21 = d2.get_beta(1)
        d211 = d21.get_beta(1)
        N1 = d.get_node()
        N2 = d2.get_node()
        N3 = d11.get_node()
        N4 = d211.get_node()
        F = d.get_face()
        F2 = d2.get_face()

        d.set_beta(1,d211)
        d2.set_beta(1,d11)
        d21.set_beta(1,d2)
        d1.set_beta(1,d)
        d11.set_beta(1,d21)
        d211.set_beta(1,d1)

        if N1.get_dart().id==d.id:
            N1.set_dart(d21)
        if N2.get_dart().id==d2.id:
            N2.set_dart(d1)

        if F.get_dart().id==d11.id:
            F.set_dart(d)

        if F2.get_dart().id == d211.id:
            F2.set_dart(d2)

        d.set_node(N3)
        d2.set_node(N4)
        d211.set_face(F)
        d11.set_face(F2)
    def add_dart(self, a1: int = -1, a2: int = -1, v: int = -1, f: int = -1) -> Dart:
        """
        This function add a dart in the mesh. It must not be used directly
        :param a1: dart index to connect by alpha1
        :param a2: dart index to connect by alpha2
        :param v: vertex index this dart points to
        :param f: face index this dart belongs to
        :return: the created dart
        """
        self.darts = numpy.append(self.darts, [[a1, a2, v, f]], axis=0)
        return Dart(self, len(self.darts))
