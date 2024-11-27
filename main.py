import sys

from user_game import user_game
#from train import train
from exploit import exploit
#from mesh_model.reader import read_gmsh
#from mesh_display import MeshDisplay


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) == 2:
        user_game(int(sys.argv[1]))
    else:
        exploit()

"""
        cmap = read_gmsh("mesh_files/irr_losange.msh")
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()
"""