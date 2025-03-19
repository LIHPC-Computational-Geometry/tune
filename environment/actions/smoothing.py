from __future__ import annotations

from mesh_model.mesh_struct.mesh import Mesh
from mesh_model.mesh_struct.mesh_elements import Node
from mesh_model.mesh_analysis.global_mesh_analysis import adjacent_darts, on_boundary

def smoothing_mean(mesh: Mesh) -> True:
    for i in range (20):
        #plot_mesh(mesh)
        for i, n_info in enumerate (mesh.nodes, start=0):
            if n_info[2] >=0:
                node_to_smooth = Node(mesh, i)
                if not on_boundary(node_to_smooth):
                    list_darts = adjacent_darts(node_to_smooth)
                    sum_x = 0.0
                    sum_y = 0.0
                    nb_nodes = 0.0
                    for d in list_darts:
                        n = d.get_node()
                        if n != node_to_smooth:
                            sum_x += n.x()
                            sum_y += n.y()
                            nb_nodes += 1
                    node_to_smooth.set_xy(sum_x/nb_nodes, sum_y/nb_nodes)