import numpy as np
import copy
import torch
import random
from tqdm import tqdm
from model_RL.actor_critic import NaNExceptionActor, NaNExceptionCritic


def train(env, actor, critic, rollouts, dataset, nb_epochs, batch_size):
    num_samples = len(dataset)
    print('Training on {}'.format(num_samples))
    for epoch in range(nb_epochs):
        start = 0
        dataset_rd = random.sample(dataset, num_samples)
        while start < num_samples - 2:
            stop = min(num_samples, start + batch_size)
            batch = dataset_rd[start:stop]
            critic_loss = []
            actor_loss = []
            critic.optimizer.zero_grad()
            for i, (s, a, r, next_s, I, done) in enumerate(batch, 1):
                X, indices_faces = env.get_x(s, None)
                X = torch.tensor(X, dtype=torch.float32)
                next_X, next_indices_faces = env.get_x(next_s, None)
                next_X = torch.tensor(next_X, dtype=torch.float32)
                value = critic(X)
                pmf = actor.forward(X)
                log_prob = torch.log(pmf[a[0]])
                next_value = torch.tensor(0.0, dtype=torch.float32) if done else critic(next_X)
                delta = r + 0.9 * next_value - value
                critic_loss.append(critic.update(delta.detach(), value))
                actor_loss.append(-log_prob * delta.detach() * I)
            actor_loss = torch.stack(actor_loss).sum()
            #critic_loss = torch.stack(critic_loss).sum()
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

            start = stop + 1

def reinforce_actor_critic(actor, critic, env):
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
    len_ep = []
    nb_iterations = 2
    nb_epochs = 5
    nb_episodes_per_rollout = 100
    batch_size = 64
    training_set = []

    try:
        for iteration in tqdm(range(nb_iterations)):
            print('ITERATION', iteration)
            rollouts=[]
            dataset = []
            for ep in range(nb_episodes_per_rollout):
                env.reset()
                trajectory = []
                ep_reward = 0
                I = 1
                done = False

                while True:
                    state = copy.deepcopy(env.mesh)
                    action = actor.select_action(state)
                    env.step(action)
                    next_state = copy.deepcopy(env.mesh)
                    R = env.reward
                    ep_reward += env.reward
                    if env.terminal:
                        if env.won:
                            wins.append(1)
                            done = True
                            trajectory.append((state, action, R, next_state, I, done))
                        else:
                            wins.append(0)
                            trajectory.append((state, action, R, next_state, I, done))
                        break
                    trajectory.append((state, action, R, next_state, I, done))
                    I = 0.9 * I
                rewards.append(ep_reward)
                rollouts.append(trajectory)
                dataset.extend(trajectory)
                len_ep.append(len(trajectory))

            train(env, actor, critic, rollouts, dataset, nb_epochs, batch_size)


    except NaNExceptionActor as e:
        print("NaN Exception on Actor Network")
        return None, None, None, None
    except NaNExceptionCritic as e:
        print("NaN Exception on Critic Network")
        return None, None, None, None
    return rewards, actor, wins, len_ep
