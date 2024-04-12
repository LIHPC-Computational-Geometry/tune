from __future__ import annotations


class Dart:
    _mesh_type: type = None

    def __init__(self, m: _mesh_type, dart_id: int):
        """
        A dart is defined from the mesh_struct it belongs to, and an id, which is the index
        in the mesh_struct container where the dart is stored. In this container, the dart d may be
        defined by 5 fields:
        - the id of the dart
        - the id of the beta1(d),
        - the id of beta2(d)
        - the id of the node d is coming from
        - the id of the face d it belongs to
        When a dart is created, it doesn't know any node or face
        :param m: a mesh_struct
        :param id: a index, that corresponds to the location of the dart data in the mesh_struct dart container
        """
        self.mesh = m
        self.id = dart_id

    def __eq__(self, a_dart: Dart) -> bool:
        """
        Equality operator between two darts. It is only based on the mesh_struct and dart info
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
    _mesh_type: type = None

    def __init__(self, m: _mesh_type, node_id: int):
        """
        A node is defined by the mesh_struct it belongs to and its id in this
        mesh_struct. Node data are stored in an array owned by its mesh_struct. A node
        can be understood as a handler to access data in a more abstract
        way
        :param m: mesh_struct the node belongs to
        :param id: node id
        """
        self.mesh = m
        self.id = node_id

    def __eq__(self, a_node: Node) -> bool:
        """
        Equality operator between two nodes. It is only based on the mesh_struct and node info and
        not on the node coordinate
        :param a_node: another node
        :return: true if the nodes are equal, false otherwise
        """
        return self.mesh == a_node.mesh and self.id == a_node.id

    def set_dart(self, dart: Dart) -> None:
        """
        Update the dart value associated with this node
        :param dart_index: the index of the dart in self.mesh_struct
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
    _mesh_type: type = None

    def __init__(self, m: _mesh_type, face_id: int):
        """
        A face is defined by the mesh_struct it belongs to and its id in this
        mesh_struct. Face data are stored in an array owned by its mesh_struct. A face
        can be understood as a handler to access data in a more abstract
        way
        :param m: mesh_struct the node belongs to
        :param id: face id
        """
        self.mesh = m
        self.id = face_id

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
        Returns the unique dart of the mesh_struct, the face point to
        :return: a dart
        """
        return Dart(self.mesh, self.mesh.faces[self.id])

    def set_dart(self, d: Dart) -> None:
        """
        Update the dart the face is defined by in the mesh_struct
        :param d: a new dart to connect to
        """
        if d is None:
            raise ValueError("Try to connect a face to a non-existing dart")
        self.mesh.faces[self.id] = d.id