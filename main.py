from view.window import Game
from environment.trimesh_env import TriMesh
from model_RL.actor_critic_epoch import Actor, Critic
from model_RL.nnPolicy import NNPolicy
from plots.create_plots import plot_average_learning_process
from mesh_display import MeshDisplay
import model.random_trimesh as TM
from actions.triangular_actions import flip_edge_ids
import sys
import json

LOCAL_MESH_FEAT = 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mesh_size = 12
    nb_runs = 1
    nb_iterations = 1
    nb_episodes = 1000
    nb_episode_per_rollout = 100
    nb_epochs = 1
    batch_size = 16
    alpha = 0.0001
    gamma = 0.9
    baseline = False
    feature = LOCAL_MESH_FEAT
    norm = True

    mesh = TM.random_flip_mesh(mesh_size)
    mesh_disp = MeshDisplay(mesh)
    #g = Game(mesh, mesh_disp)
    #g.run()

    env = TriMesh(mesh_size, feature)

    # Choix de la politique Actor Critic
    actor = Actor(env, 30, 5, lr=0.0001)
    critic = Critic(30, lr=0.0001)
    #policy = NNPolicy(env, 30, 64,5, 0.9, lr=0.0001)

    plot_average_learning_process(nb_runs, actor, critic, env, alpha)

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
