from view.window import Game
from environment.trimesh_env import TriMesh
from model_RL.actor_critic import Actor, Critic
from plots.create_plots import plot_average_learning_process
from mesh_display import MeshDisplay
import model.random_trimesh as TM
from actions.triangular_actions import flip_edge_ids
import sys
import json


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nb_episodes = 500
    runs = 1
    alpha = 0.0001
    gamma = 0.9
    baseline = False
    feature = 0
    norm = True

    mesh = TM.random_mesh(14)
    mesh_disp = MeshDisplay(mesh)
    #g = Game(mesh, mesh_disp)
    #g.run()

    env = TriMesh(mesh, feature)

    # Choix de la politique Actor Critic
    actor = Actor(env, 60, 10, lr=0.0001)
    critic = Critic(60, lr=0.0001)

    plot_average_learning_process(runs, actor, critic, env, nb_episodes, alpha)

    """
    if len(sys.argv) != 2:
        print("Usage: main.py <nb_nodes_of_the_mesh>")
    else:
        cmap = TM.random_mesh(int(sys.argv[1]))
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()

    
    #Code to load a json file and create a mesh
    
    if len(sys.argv) != 2:
        print("Usage: main.py <mesh_file.json>")
    else:
        #f = open(sys.argv[1])
        #json_mesh = json.load(f)
        #cmap = Mesh(json_mesh['nodes'], json_mesh['faces'])
        cmap = regular_mesh(int(sys.argv[1]))
        mesh_disp = MeshDisplay(cmap)
        g = Game(cmap, mesh_disp)
        g.run()
    """
