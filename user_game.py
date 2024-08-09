from view.window import Game

import model.random_trimesh as TM
from mesh_display import MeshDisplay


def user_game(mesh_size):
    # Code to load a json file and create a mesh
    """
    if len(sys.argv) != 2:
        print("Usage: main.py <mesh_file.json>")
    else:
        # f = open(sys.argv[1])
        # json_mesh = json.load(f)
        # cmap = Mesh(json_mesh['nodes'], json_mesh['faces'])
        cmap = TM.regular_mesh(int(sys.argv[1]))
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()
    """
    cmap = TM.random_mesh(mesh_size)
    mesh_disp = MeshDisplay(cmap)
    g = Game(cmap, mesh_disp)
    g.run()
