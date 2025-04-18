import mesh_model.random_trimesh as TM
import torch
from environment.trimesh_env import TriMesh

from view.mesh_plotter.create_plots import plot_training_results, plot_test_results
from view.mesh_plotter.mesh_plots import plot_dataset

from model_RL.evaluate_model import testPolicy

from model_RL.PPO_model import PPO
#from model_RL.SAC_model import SAC
#from model_RL.AC_model import AC

LOCAL_MESH_FEAT = 0


def train():
    mesh_size = 30
    lr = 0.0001
    gamma = 0.9
    feature = LOCAL_MESH_FEAT

    dataset = [TM.random_mesh(30) for _ in range(9)]
    plot_dataset(dataset)

    env = TriMesh(None, mesh_size, max_steps=80, feat=feature)

    # Choix de la politique Actor Critic
    # actor = Actor(env, 30, 5, lr=0.0001)
    # critic = Critic(30, lr=0.0001)
    # policy = NNPolicy(env, 30, 64,5, 0.9, lr=0.0001)

    model = PPO(env, lr, gamma, nb_iterations=3, nb_episodes_per_iteration=10, nb_epochs=2, batch_size=8)
    actor, rewards, wins, steps = model.train()
    if rewards is not None:
        plot_training_results(rewards, wins, steps)

    # torch.save(actor.state_dict(), 'policy_saved/actor_network.pth')
    avg_steps, avg_wins, avg_rewards, final_meshes = testPolicy(actor, 5, dataset, 60)

    if rewards is not None:
        plot_test_results(avg_rewards, avg_wins, avg_steps, avg_rewards)
    plot_dataset(final_meshes)
