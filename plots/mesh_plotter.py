import matplotlib.pyplot as plt
from model.mesh_struct.mesh_elements import Dart
from model.mesh_struct.mesh import Mesh
import numpy as np


def plot_mesh(mesh: Mesh) -> None:
    """
    Plot a mesh using matplotlib
    :param mesh: a Mesh
    """
    fig, ax = plt.subplots()
    subplot_mesh(mesh)
    plt.show()


def subplot_mesh(mesh: Mesh) -> None:
    """
    Plot a mesh using matplotlib for subplots with many meshes
    :param mesh: a Mesh
    """
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


def plot_dataset(dataset: list[Mesh]) -> None:
    """
    Plot all the meshes of a dataset with subplot.
    :param dataset: a list with all the meshes
    """
    nb_mesh = len(dataset)
    sqrt_mesh = np.sqrt(nb_mesh)
    if sqrt_mesh % 2 == 0:
        nb_lines = int(sqrt_mesh)
        nb_columns = int(sqrt_mesh)
    else:
        nb_lines = int(sqrt_mesh)
        nb_columns = int(sqrt_mesh) +1
    fig, ax = plt.subplots(nb_lines, nb_columns)
    for i, mesh in enumerate(dataset, 1):
        plt.subplot(nb_lines, nb_columns, i)
        subplot_mesh(mesh)
        plt.title('Mesh {}'.format(i))
    plt.tight_layout()
    plt.show()
