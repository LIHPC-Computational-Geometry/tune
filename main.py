from view.window import Game

import sys
import json
from model.mesh_struct.mesh import Mesh

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: main.py <mesh_file.json>")
    else:
        f = open(sys.argv[1])
        json_mesh = json.load(f)
        cmap = Mesh(json_mesh['nodes'], json_mesh['faces'])
        g = Game(cmap)
        g.run()
