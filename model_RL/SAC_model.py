from model_RL.utilities.actor_critic_networks import NaNExceptionActor, NaNExceptionCritic, Actor, Critic

import torch
import random
from tqdm import tqdm


class SAC:
    def __init__(self, env, lr, gamma, nb_epochs, steps_per_epoch, batch_size, update_every, update_after):
        self.env = env
        self.actor = Actor(env, 30, 5, lr=0.0001)
        self.critic = Critic(30, lr=0.0001)
        self.lr = lr
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = steps_per_epoch * nb_epochs
        self.batch_size = batch_size
        self.update_every = update_every
        self.update_after = update_after

    def train(self):
        rewards = []
        wins = []
        len_ep = []
        self.env.reset()
        trajectory = []
        rollouts = []
        done = False
        ep_reward = 0
        I = 1
        try:
            for t in tqdm(range(self.total_steps)):
                state = self.env.mesh
                action = self.actor.select_action(state)  # action= [action, dart_id]
                self.env.step(action)
                next_state = self.env.mesh
                R = self.env.reward
                ep_reward += R
                trajectory.append((state, action, R, next_state, I, done))
                if self.env.terminal:
                    rewards.append(ep_reward)
                    len_ep.append(t)
                    rollouts.extend(trajectory)
                    if self.env.won:
                        wins.append(1)
                        done = True
                        trajectory.append((state, action, R, next_state, I, done))
                        rollouts.extend(trajectory)
                    else:
                        wins.append(0)
                        trajectory.append((state, action, R, next_state, I, done))
                        rollouts.extend(trajectory)
                    self.env.reset()
                    trajectory = []
                    ep_reward = 0
                    I = 1
                I = 0.9 * I

                if t >= self.update_after and t % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = random.sample(rollouts, self.batch_size)
                        critic_loss = []
                        actor_loss = []
                        self.critic.optimizer.zero_grad()
                        for i, (s, a, r, next_s, I, done) in enumerate(batch, 1):
                            X, indices_faces = self.env.get_x(s, None)
                            X = torch.tensor(X, dtype=torch.float32)
                            next_X, next_indices_faces = self.env.get_x(next_s, None)
                            next_X = torch.tensor(next_X, dtype=torch.float32)
                            value = self.critic(X)
                            pmf = self.actor.forward(X)
                            log_prob = torch.log(pmf[a[0]])
                            next_value = torch.tensor(0.0, dtype=torch.float32) if done else self.critic(next_X)
                            delta = r + 0.9 * next_value - value
                            critic_loss.append(value * delta.detach())
                            actor_loss.append(-log_prob * delta.detach() * I)
                        actor_loss = torch.stack(actor_loss).sum()
                        critic_loss = torch.stack(critic_loss).sum()
                        critic_loss.backward()
                        self.critic.optimizer.step()
                        for p in self.critic.parameters():
                            p.requires_grad = False
                        self.actor.optimizer.zero_grad()
                        with torch.autograd.set_grad_enabled(True):
                            actor_loss.backward()
                        self.actor.optimizer.step()

                        for p in self.critic.parameters():
                            p.requires_grad = True

                if (t + 1) % self.steps_per_epoch == 0:
                    epoch = (t + 1) // self.steps_per_epoch
                    # A quoi ça sert ?

        except NaNExceptionActor as e:
            print("NaN Exception on Actor Network")
            return None, None
        except NaNExceptionCritic as e:
            print("NaN Exception on Critic Network")
            return None, None
        return self.actor, rewards, wins, len_ep
