import matplotlib.pyplot as plt
from model.mesh_struct.mesh_elements import Dart
import numpy as np


def mesh_plot(mesh):
    faces = mesh.faces
    nodes = mesh.nodes
    nodes = np.array([list[:2] for list in nodes])
    fig, ax = plt.subplots()

    for dart_id in faces:
        d1 = Dart(mesh, dart_id)
        d2 = d1.get_beta(1)
        d3 = d2.get_beta(1)
        n1 = d1.get_node()
        n2 = d2.get_node()
        n3 = d3.get_node()
        polygon = np.array([(n1.x(), n1.y()), (n2.x(),n2.y()), (n3.x(), n3.y()), (n1.x(), n1.y())])
        ax.plot(polygon[:, 0], polygon[:, 1], 'k-')

    # Tracer les sommets
    ax.plot(nodes[:, 0], nodes[:, 1], 'ro')  # 'ro' pour des points rouges

    # Annoter les sommets avec leurs indices
    #for i, (x, y) in enumerate(nodes):
        #ax.text(x, y, str(i), fontsize=12, ha='right')

    # Ajuster les axes pour une meilleure visualisation
    ax.set_aspect('equal')
    ax.grid(False)
    ax.axis('off')
    # Afficher la figure
    plt.show()