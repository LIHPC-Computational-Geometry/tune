import sys

import numpy


class Dart:
    def __init__(self, m, id: int):
        self.mesh = m
        self.id = id

    def get_alpha(self, i: int):
        """
        Get the dart connected to by alpha_i
        :param i: alpha dimension
        :return: the id of the dart alpha_i(self)
        """
        if i < 1 or i > 2:
            raise ValueError("Wrong alpha dimension")
        return self.mesh.darts[self.id - 1, i - 1]


    def set_alpha(self, i: int, id_to:int):
        if i < 1 or i > 2:
            raise ValueError("Wrong alpha dimension")
        self.mesh.darts[self.id - 1, i - 1] = id_to


    def get_vertex(self):
        return self.mesh.darts[self.id - 1, 2]

    def set_vertex(self, node_id: int):
        self.mesh.darts[self.id - 1, 2] = node_id

class Node:
    def __init__(self, m, id: int):
        self.mesh = m
        self.id = id

    def x(self):
        return self.mesh.vertices[self.id - 1, 0]

    def y(self):
        return self.mesh.vertices[self.id - 1, 1]

    def xy(self):
        return self.mesh.vertices[self.id - 1]

    def set_x(self, x: float):
        self.mesh.vertices[self.id - 1, 0] = x

    def set_y(self, y: float):
        self.mesh.vertices[self.id - 1, 1] = y

    def set_xy(self, x: float, y: float):
        self.set_x(x)
        self.set_y(y)


class Face:
    def __init__(self, m, d:Dart, id: int):
        self.mesh = m
        self.id = id
        self.dart = d

class Mesh:
    def __init__(self):
        self.vertices = numpy.empty((0, 2))
        self.faces = numpy.empty(0, dtype=int)
        self.darts = numpy.empty((0, 3), dtype=int)

    def nb_vertices(self):
        """
        :return: the number of vertices in the mesh
        """
        #We filter the vertices having the x-coordinate equals to max float
        return len(self.vertices[self.vertices[:,0]!= sys.float_info.max])

    def nb_faces(self):
        """
           :return: the number of faces in the mesh
           """
        # We filter the faces having the -1 value
        return len(self.faces[self.faces[:] != -1])

    def add_vertex(self, x: float, y: float) -> Node:
        """
        Add a vertex in the mesh
        :param x: X coordinate
        :param y: Y coordinate
        :return: the index of the created vertex
        """
        self.vertices = numpy.append(self.vertices, [[x, y]], axis=0)
        return Node(self, len(self.vertices))

    def del_vertex(self,i:int):
        """
        Removes the vertex of id i. Warning all the darts that point
        on this vertex will be invalid (but not automatically updated)
        :param i: the vertex id
        """
        ni = Node(self,i)
        ni.set_x(sys.float_info.max)

    def add_triangle(self, v1: int, v2: int, v3: int) -> int:
        """
        Add a triangle defined by vertices of indices v1, v2, and v3.
        The triangle is created in the order of v1, v2 then v3
        An exception is raised if one of the vertices does not exist
        :param v1: first vertex
        :param v2: second vertex
        :param v3: third vertex
        :return: the dart of the triangle that connect v3 to v1
        """
        d1 = self.add_dart()
        d2 = self.add_dart()
        d3 = self.add_dart()
        d1.set_alpha(1, d2.id)
        d2.set_alpha(1, d3.id)
        d3.set_alpha(1, d1.id)
        d1.set_vertex(v1.id)
        d2.set_vertex(v2.id)
        d3.set_vertex(v3.id)
        self.faces = numpy.append(self.faces, [d1])


    def add_dart(self, a1: int = 0, a2: int = 0, v: int = 0) -> int:
        """
        This function add a dart in the mesh. It must not be used directly
        :param a1: dart index to connect by alpha1
        :param a2: dart index to connect by alpha2
        :param v:  vertex this dart point to
        :return: the index of the created dart
        """
        self.darts = numpy.append(self.darts, [[a1, a2, v]], axis=0)
        return Dart(self,len(self.darts))
