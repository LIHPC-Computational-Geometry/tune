import numpy as np
import torch
from tqdm import tqdm
from model_RL.actor_critic import NaNExceptionActor, NaNExceptionCritic
import random


def reinforce_actor_critic(actor, critic, env, epochs, batch_size=5, steps_per_epoch=500):
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
    total_steps = steps_per_epoch * epochs
    update_after = 100
    update_every = 8
    env.reset()
    trajectory = []
    rollouts = []
    ep_reward = 0
    I = 1
    try:
        for t in tqdm(range(total_steps)):
            state = env.mesh
            action = actor.select_action(state)
            log_prob = action[2]
            action = action[:2]
            env.step(action)
            next_state = env.mesh
            R = env.reward
            ep_reward += env.reward
            if env.terminal:
                rewards.append(ep_reward)
                #rollouts.append(trajectory)
                if env.won:
                    wins.append(1)
                    rollouts.append((state, action, R, next_state, I, log_prob, True))
                else:
                    wins.append(0)
                    rollouts.append((state, action, R, next_state, I, log_prob, False))
                env.reset()
                trajectory = []
                ep_reward = 0
                I = 1

            else :
                rollouts.append((state, action, R, next_state, I, log_prob, False))
            I = 0.9 * I

            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = random.sample(rollouts, batch_size)

                    critic_loss = []
                    actor_loss = []

                    critic.optimizer.zero_grad()
                    for i, (s, a, r, next_s, I, log_prob, done) in enumerate(batch, 1):
                        with torch.no_grad():
                            X, indices_faces = env.get_x(s, None)
                            X = torch.tensor(X, dtype=torch.float32)
                            next_X, next_indices_faces = env.get_x(next_s, None)
                            next_X = torch.tensor(next_X, dtype=torch.float32)
                            value = critic(X)
                            next_value = torch.tensor(0.0, dtype=torch.float32) if done else critic(next_X)
                            delta = r + 0.9 * next_value - value
                        critic_loss.append(critic.update(delta.detach(), value))
                        actor_loss.append(-log_prob * delta.detach() * I)
                    actor_loss = torch.stack(actor_loss).sum()
                    critic_loss = torch.stack(critic_loss).sum()
                    critic_loss.backward()
                    critic.optimizer.step()
                    for p in critic.parameters():
                        p.requires_grad = False
                    actor.optimizer.zero_grad()
                    with torch.autograd.set_grad_enabled(True):
                        actor_loss.backward()
                    actor.optimizer.step()

                    for p in critic.parameters():
                        p.requires_grad = True


            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1)//steps_per_epoch
                #A quoi ça sert ?


    except NaNExceptionActor as e:
        print("NaN Exception on Actor Network")
        return None, None
    except NaNExceptionCritic as e:
        print("NaN Exception on Critic Network")
        return None, None
    return rewards, actor, wins
