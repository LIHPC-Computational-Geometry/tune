from mesh_model.mesh_analysis.global_mesh_analysis import global_score
import copy
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from mesh_model.mesh_analysis.quadmesh_analysis import isValidAction


class NaNExceptionActor(Exception):
    pass


class NaNExceptionCritic(Exception):
    pass


class Actor(nn.Module):
    def __init__(self, env, input_dim, output_dim, lr=0.0001, eps=0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = 0.9
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.env = env
        self.eps = eps

    def reset(self, env=None):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.optimizer = Adam(self.parameters(), lr=self.optimizer.defaults['lr'], weight_decay=self.optimizer.defaults['weight_decay'])

    def select_action(self, observation, info):
        if np.random.rand() < self.eps:
            action = self.env.sample() # random choice of an action
            dart_id = self.env.darts_selected[action[1]]
            action_type = action[0]
            total_actions_possible = np.prod(self.env.action_space.nvec)
            prob = 1/total_actions_possible
            i = 0
            while not isValidAction(self.env.mesh, dart_id, action_type):
                if i > 15:
                    return None, None
                action = self.env.sample()
                dart_id = self.env.darts_selected[action[1]]
                action_type = action[0]
                i += 1
        else:
            obs = torch.tensor(observation.flatten(), dtype=torch.float32)
            pmf = self.forward(obs)
            dist = Categorical(pmf)
            action = dist.sample()
            action = action.tolist()
            prob = pmf[action]
            action_dart = int(action/3)
            action_type = action % 3
            dart_id = info["darts_selected"][action_dart]
            i = 0
            while not isValidAction(info["mesh"], dart_id, action_type):
                if i > 15:
                    return None, None
                pmf = self.forward(obs)
                dist = Categorical(pmf)
                action = dist.sample()
                action = action.tolist()
                prob = pmf[action]
                action_dart = int(action/3)
                action_type = action % 3
                dart_id = info["darts_selected"][action_dart]
                i += 1
        action_list = [action, dart_id, action_type]
        return action_list, prob

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise NaNExceptionActor("Les couches cachées renvoient nan ou infinies")
        return self.softmax(x)

    def get_pi(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32)
        pmf = self.forward(obs)
        return pmf.tolist()

    def update(self, delta, L, state, action):
        obs = self.env._get_obs()
        X = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action[0], dtype=torch.int64)
        pmf = self.forward(X)
        log_prob = torch.log(pmf[action])
        actor_loss = -log_prob * delta * L
        return actor_loss

    def learn(self, actor_loss ):
        self.optimizer.zero_grad()
        actor_loss = torch.stack(actor_loss).sum()
        actor_loss.backward()
        self.optimizer.step()


class Critic(nn.Module):
    def __init__(self, input_dim, lr=0.0001):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.optimizer = Adam(self.parameters(), lr=lr)

    def reset(self, env=None):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.optimizer = Adam(self.parameters(), lr=self.optimizer.defaults['lr'])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise NaNExceptionCritic("Les couches cachées critic renvoie nan ou inf")
        return x

    def update(self, delta, value):
        critic_loss = delta * value
        return critic_loss

    def learn(self, critic_loss):
        self.optimizer.zero_grad()
        critic_loss = torch.stack(critic_loss).sum()
        critic_loss.backward()
        self.optimizer.step()


class PPO:
    def __init__(self, env, lr, gamma, nb_iterations, nb_episodes_per_iteration, nb_epochs, batch_size):
        self.env = env
        self.actor = Actor(env, 10*8, 3*10, lr=0.0001)
        self.critic = Critic(8*10, lr=0.0001)
        self.lr = lr
        self.gamma = gamma
        self.nb_iterations = nb_iterations
        self.nb_episodes_per_iteration = nb_episodes_per_iteration
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.epsilon = 0.2

    # def _setup_model(self):

    # def collect_rollout(self):

    def train(self, dataset):
        num_samples = len(dataset)
        print('training on {}'.format(num_samples))
        for _ in range(self.nb_epochs):
            start = 0
            dataset_rd = random.sample(dataset, num_samples)
            while start < num_samples - 2:
                stop = min(num_samples, start + self.batch_size)
                batch = dataset_rd[start:stop]
                critic_loss = []
                actor_loss = []
                self.critic.optimizer.zero_grad()
                G = 0
                for _, (s, o, a, r, old_prob, next_o, done) in enumerate(batch, 1):
                    o = torch.tensor(o.flatten(), dtype=torch.float32)
                    next_o = torch.tensor(next_o.flatten(), dtype=torch.float32)
                    value = self.critic(o)
                    pmf = self.actor.forward(o)
                    log_prob = torch.log(pmf[a[0]])
                    next_value = torch.tensor(0.0, dtype=torch.float32) if done else self.critic(next_o)
                    delta = r + 0.9 * next_value - value
                    G = (r + 0.9 * G) / 10
                    _, st, ideal_s, _ = global_score(s) # Comparaison à l'état s et pas s+1 ?
                    if st == ideal_s:
                        continue
                    advantage = 1 if done else G / (st - ideal_s)
                    ratio = torch.exp(log_prob - torch.log(old_prob).detach())
                    actor_loss1 = advantage * ratio
                    actor_loss2 = advantage * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    clipped_obj = min(actor_loss1, actor_loss2)
                    critic_loss.append(delta.detach() * value)
                    actor_loss.append(-clipped_obj)
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
                start = stop + 1

    def learn(self):
        """
        Train the PPO mesh_model
        :return: the actor policy, training rewards, training wins, len of episodes
        """
        rewards = []
        wins = []
        len_ep = []

        try:
            for iteration in tqdm(range(self.nb_iterations)):
                print('ITERATION', iteration)
                rollouts = []
                dataset = []
                for _ in tqdm(range(self.nb_episodes_per_iteration)):
                    next_obs, info = self.env.reset()
                    trajectory = []
                    ep_reward = 0
                    done = False
                    step = 0
                    while step < 40:
                        state = copy.deepcopy(info["mesh"])
                        obs = next_obs
                        action, prob = self.actor.select_action(obs, info)
                        if action is None:
                            wins.append(0)
                            break
                        gym_action = [action[2],int(action[0]/3)]
                        next_obs, reward, terminated, truncated, info = self.env.step(gym_action)
                        ep_reward += reward
                        if terminated:
                            if truncated:
                                wins.append(0)
                                trajectory.append((state, obs, action, reward, prob, next_obs, done))
                            else:
                                wins.append(1)
                                done = True
                                trajectory.append((state, obs, action, reward, prob, next_obs, done))
                            break
                        trajectory.append((state, obs, action, reward, prob, next_obs, done))
                        step += 1
                    if len(trajectory) != 0:
                        rewards.append(ep_reward)
                        rollouts.append(trajectory)
                        dataset.extend(trajectory)
                        len_ep.append(len(trajectory))

                self.train(dataset)

        except NaNExceptionActor:
            print("NaN Exception on Actor Network")
            return None, None, None, None
        except NaNExceptionCritic:
            print("NaN Exception on Critic Network")
            return None, None, None, None
        return self.actor, rewards, wins, len_ep
