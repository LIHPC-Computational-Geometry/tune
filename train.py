import model.random_trimesh as TM

from environment.trimesh_env import TriMesh
from mesh_display import MeshDisplay

from plots.create_plots import plot_training_results, plot_test_results
from plots.mesh_plotter import plot_mesh, plot_dataset

from model_RL.test_model import testPolicy

from model_RL.PPO_model import PPO
from model_RL.SAC_model import SAC
from model_RL.AC_model import AC

LOCAL_MESH_FEAT = 0


def train():
    mesh_size = 12
    nb_runs = 1
    nb_iterations = 2
    nb_episodes = 20
    nb_episode_per_iteration = 20
    nb_epochs = 1
    batch_size = 8
    lr = 0.0001
    gamma = 0.9
    baseline = False
    feature = LOCAL_MESH_FEAT
    norm = True

    mesh = TM.random_flip_mesh(mesh_size)
    mesh_disp = MeshDisplay(mesh)
    # plot_mesh(mesh)
    dataset = [TM.random_flip_mesh(30) for _ in range(16)]
    plot_dataset(dataset)

    env = TriMesh(None, mesh_size, max_steps=30)

    # Choix de la politique Actor Critic
    # actor = Actor(env, 30, 5, lr=0.0001)
    # critic = Critic(30, lr=0.0001)
    # policy = NNPolicy(env, 30, 64,5, 0.9, lr=0.0001)

    model = AC(env, lr, gamma, nb_episodes=50)
    actor, rewards, wins, steps = model.train()

    avg_steps, avg_wins, avg_rewards, final_meshes = testPolicy(actor, 10, dataset, 60)

    if rewards is not None:
        plot_training_results(rewards, wins, steps)
        plot_test_results(avg_rewards, avg_wins, avg_steps)
    plot_dataset(final_meshes)

    # plot_average_learning_process(nb_runs, actor, critic, env, alpha)