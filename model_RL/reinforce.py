import numpy as np
from tqdm import tqdm
from model_RL.nnPolicy import NaNException


def reinforce(policy, env, nb_episodes, baseline=False):
    """
    Reinforce algorithm to trained the policy
    :param policy: a policy
    :param env: an environment
    :param nb_episodes: number of episodes
    :param baseline: whether to use baseline or not
    :return: rewards, policy
    """
    rewards = []
    wins = []
    try:
        for ep in tqdm(range(nb_episodes), leave=False):
            env.reset()
            trajectory = []
            ep_reward = 0

            while True:
                state = env.mesh
                action = policy.select_action(state)
                env.step(action)
                ep_reward += env.reward
                trajectory.append((state, action, env.reward))
                if env.terminal:
                    if env.won:
                        wins.append(1)
                    else:
                        wins.append(0)
                    break
            rewards.append(ep_reward)
            if baseline:
                policy.update_w_baseline(trajectory)
            else:
                policy.update(trajectory)

    except NaNException as e:
        print("NaN exception occurred")
        return None, None, None

    return rewards, policy, wins
