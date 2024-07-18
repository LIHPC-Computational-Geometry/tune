import numpy as np
import torch
from tqdm import tqdm
from model_RL.actor_critic import NaNExceptionActor, NaNExceptionCritic


def reinforce_actor_critic(actor, critic, env, nb_episodes):
    """
    Reinforce algorithm to train the policy
    :param actor: Actor network
    :param critic: Critic network
    :param env: an environment
    :param nb_episodes: number of episodes
    :return: rewards, policy
    """
    rewards = []
    try:
        for ep in tqdm(range(nb_episodes)):
            env.reset()
            trajectory = []
            ep_reward = 0
            I = 1
            actor_loss = []
            critic_loss = []

            while True:
                state = env.mesh
                action = actor.select_action(state)
                env.step(action)
                next_state = env.mesh
                R = env.reward
                ep_reward += env.reward
                trajectory.append((state, action, env.reward))
                # Actor Critic update
                X, indices_faces = env.get_x(state, None)
                X = torch.tensor(X, dtype=torch.float32)
                next_X, next_indices_faces = env.get_x(next_state, None)
                next_X = torch.tensor(next_X, dtype=torch.float32)
                value = critic(X)
                next_value = 0 if env.terminal else critic(next_X)
                delta = R + 0.9*next_value - value
                critic_loss.append(critic.update(delta, value))
                actor_loss.append(actor.update(delta.detach(), I, state, action))
                I = 0.9*I
                if env.terminal:
                    break
            critic.learn(critic_loss)
            actor.learn(actor_loss)
            rewards.append(ep_reward)
    except NaNExceptionActor as e:
        print("NaN Exception on Actor Network")
        return None, None, None
    except NaNExceptionCritic as e:
        print("NaN Exception on Critic Network")
        return None, None, None
    return rewards, actor
