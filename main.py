from view.window import Game
from model.mesh_struct.mesh import Mesh
from mesh_display import MeshDisplay
import model.random_trimesh as TM
import sys
import json


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: main.py <nb_nodes_of_the_mesh>")
    else:
        cmap = TM.random_mesh(int(sys.argv[1]))
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()

    """
    #Code to load a json file and create a mesh
    
    if len(sys.argv) != 2:
        print("Usage: main.py <mesh_file.json>")
    else:
        f = open(sys.argv[1])
        json_mesh = json.load(f)
        cmap = Mesh(json_mesh['nodes'], json_mesh['faces'])
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()
    """
