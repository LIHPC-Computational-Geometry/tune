
from view import graph
from view.window import Game

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nodes = [(0.0, 0.0), (5.0, 5.0), (0.0, 5.0), (-2.0, 2.5)]
    edges = [(0, 1), (0, 2), (2, 1), (0, 3), (2, 3)]
    m = graph.Mesh(nodes, edges)
    g = Game(m)
    g.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
