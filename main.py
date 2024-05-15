from view.window import Game
from model.mesh_struct.mesh import Mesh
from mesh_display import MeshDisplay
from model.random_trimesh import regular_mesh, random_mesh
import sys
import json


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    cmap = random_mesh(30)
    mesh_disp = MeshDisplay(cmap)
    g = Game(cmap, mesh_disp)
    g.run()




"""
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
