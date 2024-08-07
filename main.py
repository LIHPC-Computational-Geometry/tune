from view.window import Game
from environment.trimesh_env import TriMesh
from model_RL.actor_critic_networks import Actor, Critic
from model_RL.nnPolicy import NNPolicy
from model_RL.PPO_model import PPO
from plots.create_plots import plot_average_learning_process, plot_training, plot_test
from plots.mesh_plotter import plot_mesh, plot_dataset
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
    nb_iterations = 7
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
    #plot_mesh(mesh)
    #g = Game(mesh, mesh_disp)
    #g.run()
    dataset = [TM.random_flip_mesh(30) for _ in range(16)]
    plot_dataset(dataset)

    env = TriMesh(None, mesh_size, max_steps=30)

    # Choix de la politique Actor Critic
    #actor = Actor(env, 30, 5, lr=0.0001)
    #critic = Critic(30, lr=0.0001)
    #policy = NNPolicy(env, 30, 64,5, 0.9, lr=0.0001)

    model = PPO(env, lr, gamma, nb_iterations, nb_episode_per_iteration, nb_epochs, batch_size)
    actor, rewards, wins, steps = model.train()
    
    avg_steps, avg_wins, avg_rewards, final_meshes = testPolicy(actor, 10, dataset, 60)

    if rewards is not None:
        plot_training(rewards, wins, steps)
        plot_test(avg_rewards, avg_wins, avg_steps)
    plot_dataset(final_meshes)

    #plot_average_learning_process(nb_runs, actor, critic, env, alpha)

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
