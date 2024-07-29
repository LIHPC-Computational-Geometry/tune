import numpy as np
import torch
from tqdm import tqdm
from model_RL.actor_critic_epoch import NaNExceptionActor, NaNExceptionCritic


def reinforce_actor_critic(actor, critic, env, nb_epochs, nb_episodes, batch_size):
    """
    Reinforce algorithm to train the policy
    :param actor: Actor network
    :param critic: Critic network
    :param env: an environment
    :param nb_episodes: number of episodes
    :return: rewards, policy
    """
    rewards = []
    wins = []
    try:
        for epoch in tqdm(range(nb_epochs)):
            rollouts=[]
            for ep in tqdm(range(nb_episodes)):
                env.reset()
                trajectory = []
                ep_reward = 0
                I = 1

                while True:
                    state = env.mesh
                    action = actor.select_action(state)
                    log_prob = action[2]
                    action = action[:2]
                    env.step(action)
                    next_state = env.mesh
                    R = env.reward
                    ep_reward += env.reward
                    trajectory.append((state, action, R, next_state, I, log_prob))
                    I = 0.9 * I
                    if env.terminal:
                        if env.won:
                            wins.append(1)
                        else:
                            wins.append(0)
                        break

                rewards.append(ep_reward)
                rollouts.append(trajectory)

            for rollout in rollouts:
                for num_batch in range(0, len(trajectory), batch_size):
                    batch = trajectory[num_batch:num_batch+batch_size]
                    critic_loss = []
                    actor_loss = []
                    for i, (s, a, r, next_s, I, log_prob) in enumerate(reversed(batch), 1):
                        X, indices_faces = env.get_x(s, None)
                        X = torch.tensor(X, dtype=torch.float32)
                        next_X, next_indices_faces = env.get_x(next_s, None)
                        next_X = torch.tensor(next_X, dtype=torch.float32)
                        value = critic(X)
                        with torch.no_grad():
                            next_value = torch.tensor(0.0, dtype=torch.float32) if num_batch==len(trajectory) else critic(next_X)
                        delta = r + 0.9 * next_value - value
                        critic_loss.append(critic.update(delta.detach(), value))
                        actor_loss.append(-log_prob * delta.detach() * I)
                    actor_loss = torch.stack(actor_loss).sum()
                    actor.learn(actor_loss)
                    critic_loss = torch.stack(critic_loss).sum()
                    critic.learn(critic_loss)

    except NaNExceptionActor as e:
        print("NaN Exception on Actor Network")
        return None, None
    except NaNExceptionCritic as e:
        print("NaN Exception on Critic Network")
        return None, None
    return rewards, actor, wins
