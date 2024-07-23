from model_RL.reinforce_actor_critic import reinforce_actor_critic
from model_RL.reinforce import reinforce
from model_RL.actor_critic import Actor, Critic

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# possible actions
FLIP = 0

LOCAL = 0
PARTIAL = 1


def plot_average_learning_process(runs: int, actor, critic, env, nb_episodes, alpha):
    avg_run_rewards = np.zeros(nb_episodes)
    real_runs = 0
    avg_wins = np.zeros(nb_episodes)

    for r in tqdm(range(runs)):
        if critic is not None:
            critic.reset(env)
        actor.reset(env)
        rewards, policy_trained, win = reinforce_actor_critic(actor, critic, env, nb_episodes)
        if rewards is None:
            real_runs -= 1
        else:
            avg_run_rewards += (1.0 / (r + 1)) * (rewards - avg_run_rewards)
            avg_wins += win
        real_runs += 1
    plt.figure()
    plt.plot(avg_run_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Learning Rewards, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()
    plt.figure()
    plt.plot(avg_wins)
    plt.xlabel('Episodes')
    plt.ylabel('Wins')
    plt.title('Learning Wins, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()