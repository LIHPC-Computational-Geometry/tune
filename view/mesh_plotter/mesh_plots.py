import matplotlib.pyplot as plt
from mesh_model.mesh_struct.mesh_elements import Dart
from mesh_model.mesh_struct.mesh import Mesh
import numpy as np

from mesh_model.reader import read_gmsh


def plot_mesh(mesh: Mesh, debug=False) -> None:
    """
    Plot a mesh using matplotlib
    :param mesh: a Mesh
    :param debug: debug mode to plot darts ID and nodes ID
    """
    fig, ax = plt.subplots(figsize=(15, 15))

    subplot_mesh(mesh, debug=debug)
    plt.show(block=True)


def subplot_mesh(mesh: Mesh, debug=False) -> None:
    """
    Plot a mesh using matplotlib for subplots with many meshes
    :param mesh: a Mesh
    """
    faces = mesh.active_faces()
    nodes = mesh.active_nodes()
    nodes = np.array([list[:2] for list in nodes])

    active_darts = mesh.active_darts()
    d = Dart(mesh, active_darts[0][0])
    d1 = d.get_beta(1)
    d11 = d1.get_beta(1)
    d111 = d11.get_beta(1)
    tri, quad = False, False
    if d111 == d:
        tri=True
    else:
        quad=True
    if tri:
        for dart_id in faces:
            d1 = Dart(mesh, dart_id)
            d2 = d1.get_beta(1)
            d3 = d2.get_beta(1)
            n1 = d1.get_node()
            n2 = d2.get_node()
            n3 = d3.get_node()

            # Nodes coordinates
            p1 = np.array([n1.x(), n1.y()])
            p2 = np.array([n2.x(), n2.y()])
            p3 = np.array([n3.x(), n3.y()])

            polygon = np.array([(n1.x(), n1.y()), (n2.x(), n2.y()), (n3.x(), n3.y()), (n1.x(), n1.y())])
            plt.plot(polygon[:, 0], polygon[:, 1], 'k-')

            if debug:
                # Plot darts ID
                mid1 = (p1 + p2) / 2
                mid2 = (p2 + p3) / 2
                mid3 = (p3 + p1) / 2

                centroid = (p1 + p2 + p3) / 3

                pos1 = mid1 +0.2* (centroid - mid1)
                pos2 = mid2 +0.2* (centroid - mid2)
                pos3 = mid3 +0.2* (centroid - mid3)

                plt.text(*pos1, f"{d1.id}", color='blue', fontsize=10, ha='center', va='center')
                plt.text(*pos2, f"{d2.id}", color='blue', fontsize=10, ha='center', va='center')
                plt.text(*pos3, f"{d3.id}", color='blue', fontsize=10, ha='center', va='center')

        if debug:
            # Plot nodes ID
            n_id =0
            for n_info in mesh.nodes:
                if n_info[2] >=0:
                    plt.text(n_info[0] + 0.03, n_info[1] - 0.02, f"{n_id}", fontsize=12, color='red', ha='right', va='top')
                n_id+=1

    elif quad:
        for dart_id in faces:
            d1 = Dart(mesh, dart_id)
            d2 = d1.get_beta(1)
            d3 = d2.get_beta(1)
            d4 = d3.get_beta(1)
            n1 = d1.get_node()
            n2 = d2.get_node()
            n3 = d3.get_node()
            n4 = d4.get_node()
            polygon = np.array([(n1.x(), n1.y()), (n2.x(), n2.y()), (n3.x(), n3.y()), (n4.x(), n4.y()), (n1.x(), n1.y())])
            plt.plot(polygon[:, 0], polygon[:, 1], 'k-')

            if debug:
                #Plot darts ID
                # Nodes coordinates
                p1 = np.array([n1.x(), n1.y()])
                p2 = np.array([n2.x(), n2.y()])
                p3 = np.array([n3.x(), n3.y()])
                p4 = np.array([n4.x(), n4.y()])

                mid1 = (p1 + p2) / 2
                mid2 = (p2 + p3) / 2
                mid3 = (p3 + p4) / 2
                mid4 = (p4 + p1) / 2

                centroid = (p1 + p2 + p3 + p4) / 4

                pos1 = mid1 + 0.2 * (centroid - mid1)
                pos2 = mid2 + 0.2 * (centroid - mid2)
                pos3 = mid3 + 0.2 * (centroid - mid3)
                pos4 = mid4 + 0.2 * (centroid - mid4)

                plt.text(*pos1, f"{d1.id}", color='blue', fontsize=10, ha='center', va='center')
                plt.text(*pos2, f"{d2.id}", color='blue', fontsize=10, ha='center', va='center')
                plt.text(*pos3, f"{d3.id}", color='blue', fontsize=10, ha='center', va='center')
                plt.text(*pos4, f"{d4.id}", color='blue', fontsize=10, ha='center', va='center')
        if debug :
            # Plot nodes ID
            n_id = 0
            for n_info in mesh.nodes:
                if n_info[2] >= 0:
                    plt.text(n_info[0] + 0.03, n_info[1] - 0.02, f"{n_id}", fontsize=12, color='red', ha='right',
                                 va='top')
                n_id += 1
    else:
        raise NotImplementedError

    # Tracer les sommets
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro')  # 'ro' pour des points rouges
    plt.grid(False)
    plt.axis('off')


def plot_dataset(dataset: list[Mesh]) -> None:
    """
    Plot all the meshes of a dataset with subplot.
    :param dataset: a list with all the meshes
    """
    nb_mesh = len(dataset)
    sqrt_mesh = np.sqrt(nb_mesh)
    if float(sqrt_mesh).is_integer():
        nb_lines = int(sqrt_mesh)
        nb_columns = int(sqrt_mesh)
    else:
        nb_lines = round(sqrt_mesh)
        nb_columns = int(sqrt_mesh) +1
    _, _ = plt.subplots(nb_lines, nb_columns, figsize=(20,20))
    for i, mesh in enumerate(dataset, 1):
        plt.subplot(nb_lines, nb_columns, i)
        subplot_mesh(mesh)
        #plt.title('Mesh {}'.format(i))
    plt.tight_layout()
    plt.show()


def dataset_plt(dataset: list[Mesh]):
    """
    Plot all the meshes of a dataset with subplot.
    :param dataset: a list with all the meshes
    """
    nb_mesh = len(dataset)
    sqrt_mesh = np.sqrt(nb_mesh)
    if float(sqrt_mesh).is_integer():
        nb_lines = int(sqrt_mesh)
        nb_columns = int(sqrt_mesh)
    else:
        nb_lines = round(sqrt_mesh)
        nb_columns = int(sqrt_mesh) +1
    fig, _ = plt.subplots(nb_lines, nb_columns)
    for i, mesh in enumerate(dataset, 1):
        plt.subplot(nb_lines, nb_columns, i)
        subplot_mesh(mesh)
        plt.title('Mesh {}'.format(i))
    plt.tight_layout()
    return fig
