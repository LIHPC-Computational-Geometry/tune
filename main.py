from view.window import Game
import model.Linear2CMap

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    faces = [(0, 1, 2), (0, 2, 3)]

    cmap = model.Linear2CMap.Mesh(nodes, faces)

#TODO Remove the mesh from the main.
# The game is built from the cmap directly
    g = Game(cmap)
    g.run()
