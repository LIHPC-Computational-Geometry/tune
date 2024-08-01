from model_RL.reinforce_actor_critic_PPO import reinforce_actor_critic
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


def plot_average_learning_process(runs: int, actor, critic, env, alpha):
    real_runs = 0
    list_rewards = []
    list_wins = []
    list_len = []

    for r in tqdm(range(runs)):
        print(f'Running run {r}')
        if critic is not None:
            critic.reset(env)
        actor.reset(env)
        rewards, policy_trained, win, len_ep = reinforce_actor_critic(actor, critic, env)
        list_rewards.append(rewards)
        list_wins.append(win)
        list_len.append(len_ep)
        """
        if rewards is None:
            real_runs -= 1
        else:
            avg_run_rewards += (1.0 / (r + 1)) * (rewards - avg_run_rewards)
            avg_wins += win"""
        real_runs += 1
    nb_episodes = min(len(lst) for lst in list_rewards)
    avg_run_rewards = np.zeros(nb_episodes)
    avg_wins = np.zeros(nb_episodes)
    avg_len = np.zeros(nb_episodes)

    for lst in list_rewards:
        for i in range(nb_episodes):
            avg_run_rewards[i] += lst[i]
    avg_run_rewards /= runs

    for lst in list_wins:
        for i in range(nb_episodes):
            avg_wins[i] += lst[i]

    for lst in list_len:
        for i in range(nb_episodes):
            avg_len[i] += lst[i]
    avg_len /= runs

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
    plt.figure()
    plt.plot(avg_len)
    plt.xlabel('Episodes')
    plt.ylabel('Number of steps per episode')
    plt.title('Learning steps, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()