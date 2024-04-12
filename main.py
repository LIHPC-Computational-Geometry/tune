from Mesh_display import MeshDisplay
from view.window import Game

import sys
import json
from model.mesh_struct.mesh import Mesh
from Mesh_display import MeshDisplay

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: main.py <mesh_file.json>")
    else:
        f = open(sys.argv[1])
        json_mesh = json.load(f)
        cmap = Mesh(json_mesh['nodes'], json_mesh['faces'])
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()
