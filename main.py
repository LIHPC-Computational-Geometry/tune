from view.window import Game
from environment.trimesh_env import TriMesh
from model_RL.actor_critic_networks import Actor, Critic
from model_RL.nnPolicy import NNPolicy
from model_RL.PPO_model import PPO
from plots.create_plots import plot_average_learning_process, plot_training, plot_test
from plots.mesh_plotter import mesh_plot
from mesh_display import MeshDisplay
import model.random_trimesh as TM
from model_RL.test_model import testPolicy
from actions.triangular_actions import flip_edge_ids
import sys
import json

LOCAL_MESH_FEAT = 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mesh_size = 12
    nb_runs = 1
    nb_iterations = 4
    nb_episodes = 100
    nb_episode_per_iteration = 100
    nb_epochs = 1
    batch_size = 8
    lr = 0.0001
    gamma = 0.9
    baseline = False
    feature = LOCAL_MESH_FEAT
    norm = True

    mesh = TM.random_flip_mesh(mesh_size)
    mesh_disp = MeshDisplay(mesh)
    mesh_plot(mesh)
    #g = Game(mesh, mesh_disp)
    #g.run()

    env = TriMesh(None, mesh_size)

    # Choix de la politique Actor Critic
    #actor = Actor(env, 30, 5, lr=0.0001)
    #critic = Critic(30, lr=0.0001)
    #policy = NNPolicy(env, 30, 64,5, 0.9, lr=0.0001)
    """
    model = PPO(env, lr, gamma, nb_iterations, nb_episode_per_iteration, nb_epochs, batch_size)
    actor, rewards, wins, steps = model.train()
    dataset = [TM.random_flip_mesh(12) for _ in range(10)]
    avg_steps, avg_wins, avg_rewards = testPolicy(actor, 10, dataset)

    if rewards is not None:
        plot_training(rewards, wins, steps)
        plot_test(avg_rewards, avg_wins, avg_steps)

    #plot_average_learning_process(nb_runs, actor, critic, env, alpha)
    """
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
