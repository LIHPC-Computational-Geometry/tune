import matplotlib.pyplot as plt
from model.mesh_struct.mesh_elements import Dart
import numpy as np


def plot_mesh(mesh):
    fig, ax = plt.subplots()
    subplot_mesh(mesh, ax)
    plt.show()

def subplot_mesh(mesh, ax):
    faces = mesh.faces
    nodes = mesh.nodes
    nodes = np.array([list[:2] for list in nodes])

    for dart_id in faces:
        d1 = Dart(mesh, dart_id)
        d2 = d1.get_beta(1)
        d3 = d2.get_beta(1)
        n1 = d1.get_node()
        n2 = d2.get_node()
        n3 = d3.get_node()
        polygon = np.array([(n1.x(), n1.y()), (n2.x(), n2.y()), (n3.x(), n3.y()), (n1.x(), n1.y())])
        plt.plot(polygon[:, 0], polygon[:, 1], 'k-')

    # Tracer les sommets
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # 'ro' pour des points rouges
    # Annoter les sommets avec leurs indices
    # for i, (x, y) in enumerate(nodes):
    # ax.text(x, y, str(i), fontsize=12, ha='right')
    plt.grid(False)
    plt.axis('off')

def plot_dataset(dataset):
    nb_mesh = len(dataset)
    sqrt_mesh = np.sqrt(nb_mesh)
    if sqrt_mesh%2 == 0:
        nb_lignes = int(sqrt_mesh)
        nb_colonnes = int(sqrt_mesh)
    else :
        nb_lignes = int(sqrt_mesh)
        nb_colonnes = int(sqrt_mesh) +1
    fig, ax = plt.subplots(nb_lignes, nb_colonnes)
    for i, mesh in enumerate(dataset, 1):
        plt.subplot(nb_lignes, nb_colonnes, i)
        subplot_mesh(mesh, ax)
        plt.title('Mesh {}'.format(i))
    plt.tight_layout()
    plt.show()
