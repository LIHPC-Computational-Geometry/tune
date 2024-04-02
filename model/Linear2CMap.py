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
        defined by 5 fields:
        - the id of the dart
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

    def __eq__(self, a_dart: Dart) -> bool:
        """
        Equality operator between two darts. It is only based on the mesh and dart info
        :param a_dart: another dart
        :return: true if the darts are equal, false otherwise
        """
        return self.mesh == a_dart.mesh and self.id == a_dart.id
    def get_beta(self, i: int) -> Dart:
        """
        Get the dart connected to by alpha_i
        :param i: alpha dimension
        :return: the id of the dart alpha_i(self)
        """
        if i < 1 or i > 2:
            raise ValueError("Wrong alpha dimension")

        if self.mesh.dart_info[self.id, i] == -1:
            return None

        return Dart(self.mesh, self.mesh.dart_info[self.id, i])

    def set_beta(self, i: int, dart_to: Dart) -> None:
        """
        Set the dart connected to by alpha_i
        :param i: dimension of the beta function
        :param dart_to: dart to connect to
        :raises ValueError: if the dimension is not valid
        """
        if i < 1 or i > 2:
            raise ValueError("Wrong alpha dimension")
        self.mesh.dart_info[self.id, i] = dart_to.id

    def get_node(self) -> Node:
        """
        :return: the embedded node of the dart. It is the node
        the dart comes from
        :raises ValueError: if there is no embedded node
        """
        node_id = self.mesh.dart_info[self.id, 3]
        if node_id == -1:
            raise ValueError("No associated node found")
        return Node(self.mesh, node_id)

    def set_node(self, node: Node) -> None:
        """
        Set the embedded node to the given value
        :param node: the embedded node
        """
        self.mesh.dart_info[self.id, 3] = node.id

    def get_face(self) -> Face:
        """
        :return: the embedded face of the dart.
        :raises ValueError: if there is no embedded face
        """
        face_id = self.mesh.dart_info[self.id, 4]
        if face_id == -1:
            raise ValueError("No associated face found")
        return Face(self.mesh, face_id)

    def set_face(self, face: Face) -> None:
        """
        Set the embedded face of the dart to the given value
        :param face: the face we want embed the dart on
        """
        self.mesh.dart_info[self.id, 4] = face.id


class Node:
    def __init__(self, m: Mesh, id: int):
        """
        A node is defined by the mesh it belongs to and its id in this
        mesh. Node data are stored in an array owned by its mesh. A node
        can be understood as a handler to access data in a more abstract
        way
        :param m: mesh the node belongs to
        :param id: node id
        """
        self.mesh = m
        self.id = id

    def __eq__(self, a_node: Node) -> bool:
        """
        Equality operator between two nodes. It is only based on the mesh and node info and
        not on the node coordinate
        :param a_node: another node
        :return: true if the nodes are equal, false otherwise
        """
        return self.mesh == a_node.mesh and self.id == a_node.id

    def set_dart(self, dart: Dart) -> None:
        """
        Update the dart value associated with this node
        :param dart_index: the index of the dart in self.mesh
        """
        if dart is None:
            raise ValueError("Try to connect a node to a non-existing dart")

        self.mesh.nodes[self.id, 2] = dart.id

    def get_dart(self) -> Dart:
        """
        Get the dart value associated with this node
        :return: a dart
        """
        return Dart(self.mesh, self.mesh.nodes[self.id, 2])

    def x(self) -> float:
        """
        Return the x coordinate of this node
        :return: the x coordinate of this node
        """
        return self.mesh.nodes[self.id, 0]

    def y(self) -> float:
        """
        Return the y coordinate of this node
        :return: the y coordinate of this node
        """
        return self.mesh.nodes[self.id, 1]

    def set_x(self, x: float) -> None:
        """
        Set the x coordinate of this node
        :param x: a float value
        """
        self.mesh.nodes[self.id, 0] = x

    def set_y(self, y: float) -> None:
        """
        Set the y coordinate of this node
        :param y: a float value
        """
        self.mesh.nodes[self.id, 1] = y

    def set_xy(self, x: float, y: float) -> None:
        """
        Set the node coordinates
        :param x: new x coordinate value
        :param y: new y coordinate value
        """
        self.set_x(x)
        self.set_y(y)


class Face:
    def __init__(self, m: Mesh, id: int):
        """
        A face is defined by the mesh it belongs to and its id in this
        mesh. Face data are stored in an array owned by its mesh. A face
        can be understood as a handler to access data in a more abstract
        way
        :param m: mesh the node belongs to
        :param id: face id
        """
        self.mesh = m
        self.id = id

    def get_nodes(self) -> list[Node]:
        """
        Gives access to the list of nodes that bounds the faces. To get
        this list of nodes, the algorithm starts from the dart owned by the face,
        and traverse the face using beta_1. For each dart, we store the corresponding
        node.
        :return: a list of nodes
        """
        start = self.get_dart()
        nodes_list = []
        do_loop = False
        while not do_loop:
            nodes_list.append(start.get_node())
            if start.get_beta(1) == self.get_dart():
                do_loop = True
            start = start.get_beta(1)


        return nodes_list

    def get_dart(self) -> Dart:
        """
        Returns the unique dart of the mesh, the face point to
        :return: a dart
        """
        return Dart(self.mesh, self.mesh.faces[self.id])

    def set_dart(self, dart: Dart) -> None:
        """
        Update the dart the face is defined by in the mesh
        :param dart: a new dart to connect to
        """
        if dart is None:
            raise ValueError("Try to connect a face to a non-existing dart")
        self.mesh.faces[self.id] = dart.id


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
        for d_info in self.dart_info:
            d = Dart(self, d_info[0])
            if d.get_beta(2) is None:
                # d is not 2-sew, we look for a dart to connect. If we don'f find one,
                # it means we are on the mesh boundary

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

    def get_nodes_coordinates(self):
        """
        Build a list containing the coordinates of the all the mesh nodes
        :return: a list of coordinates (x,y)
        """
        node_list = []
        for n in self.nodes:
            node_list.append((n[0], n[1]))
        return node_list

    def get_edges(self):
        """
        Build a list containing the coordinates of the all the mesh nodes
        :return: a list of coordinates (x,y)
        """
        edge_list = []
        for d in self.dart_info:
            n1 = Dart(self, d[0]).get_node()
            n2 = Dart(self, d[1]).get_node()
            if (d[2] != -1 and n1.id < n2.id) or d[2] == -1:
                edge_list.append((n1.id, n2.id))
        return edge_list

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
        tri = Face(self, len(self.faces)-1)

        d1.set_face(tri)
        d2.set_face(tri)
        d3.set_face(tri)

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
        # create darts
        d1 = self.add_dart()
        d2 = self.add_dart()
        d3 = self.add_dart()
        d4 = self.add_dart()

        d1.set_beta(1, d2)
        d2.set_beta(1, d3)
        d3.set_beta(1, d4)
        d4.set_beta(1, d1)

        d1.set_node(n1)
        d2.set_node(n2)
        d3.set_node(n3)
        d4.set_node(n4)

        n1.set_dart(d1)
        n2.set_dart(d2)
        n3.set_dart(d3)
        n4.set_dart(d4)

        self.faces = numpy.append(self.faces, [d1.id])
        quad = Face(self, len(self.faces)-1)

        d1.set_face(quad)
        d2.set_face(quad)
        d3.set_face(quad)
        d4.set_face(quad)

        return quad

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

    def flip_edge_ids(self, id1: int, id2: int) -> bool:
        return self.flip_edge(Node(self, id1), Node(self, id2))

    def flip_edge(self, n1: Node, n2: Node) -> bool:
        found, d = self.find_inner_edge(n1, n2)
        if not found:
            return False
        d2 = d.get_beta(2)
        d1 = d.get_beta(1)
        d11 = d1.get_beta(1)
        d21 = d2.get_beta(1)
        d211 = d21.get_beta(1)
        n1 = d.get_node()
        n2 = d2.get_node()
        n3 = d11.get_node()
        n4 = d211.get_node()
        f1 = d.get_face()
        f2 = d2.get_face()

        d.set_beta(1, d211)
        d2.set_beta(1, d11)
        d21.set_beta(1, d2)
        d1.set_beta(1, d)
        d11.set_beta(1, d21)
        d211.set_beta(1, d1)

        if n1.get_dart().id == d.id:
            n1.set_dart(d21)
        if n2.get_dart().id == d2.id:
            n2.set_dart(d1)

        if f1.get_dart().id == d11.id:
            f1.set_dart(d)

        if f2.get_dart().id == d211.id:
            f2.set_dart(d2)

        d.set_node(n3)
        d2.set_node(n4)
        d211.set_face(f1)
        d11.set_face(f2)
        return True

    def set_face_beta2(self, f: Face, darts: (Dart)):
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
            end = (df_current.id != f.get_dart().id)

    def split_edge_ids(self, id1: int, id2: int) -> bool:
        return self.split_edge(Node(self,id1), Node(self,id2))

    def split_edge(self, n1: Node, n2: Node) -> bool:
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

        # create a new node in the middle of [n1, n2]
        N5 = self.add_node( (N1.x() + N2.x()) / 2, (N1.y() + N2.y()) / 2)

        # modify existing triangles
        print("Avant modif triangle")
        print(self.nodes)
        d1.set_node(N5)
        d21.set_node(N5)

        # create 2 new triangles
        print("Avant ajout de triangles")
        F3 = self.add_triangle(N5, N1, N2)
        F4 = self.add_triangle(N5, N3, N4)

        # update beta2 relations
        self.set_face_beta2(F3, (d, d21))
        self.set_face_beta2(F4, (d1, d2))
        return True

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
