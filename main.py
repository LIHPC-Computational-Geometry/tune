from Mesh_display import MeshDisplay
from view.window import Game

import sys
import json
import model.Linear2CMap
import Mesh_display

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: main.py <mesh_file.json>")
    else:
        f = open(sys.argv[1])
        json_mesh = json.load(f)
        cmap = model.Linear2CMap.Mesh(json_mesh['nodes'], json_mesh['faces'])
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()
