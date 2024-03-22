
from view import graph
from view.window import Game
import model.Linear2CMap

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3)]
    cmap =model.Linear2CMap.Mesh(nodes,edges)
    m = graph.Mesh(nodes, edges)
    g = Game(m)
    g.run()

