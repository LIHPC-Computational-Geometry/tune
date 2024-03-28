from view.window import Game

import sys
import json
import model.Linear2CMap

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: main.py <mesh_file.json>")
    else:
        f = open(sys.argv[1])
        json_mesh = json.load(f)
        nodes = json_mesh['nodes']
        faces = json_mesh['faces']
        print("Nodes: " + str(nodes))
        print("Faces: " + str(faces))

        cmap = model.Linear2CMap.Mesh(nodes, faces)

        #TODO Remove the mesh from the main.
        # The game is built from the cmap directly
        g = Game(cmap)
        g.run()
