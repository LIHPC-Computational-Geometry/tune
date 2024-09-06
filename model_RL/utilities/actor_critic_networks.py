import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from model.mesh_analysis import isValidAction


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

    def select_action(self, state):
        if np.random.rand() < self.eps:
            X, dart_indices = self.env.get_x(state, None)
            action = np.random.randint(5*3) # random choice of 3 actions on 3 darts
            dart_id = dart_indices[int(action/3)]
            action_type = action % 3
            prob = 1/3
            i = 0
            while not isValidAction(state, dart_id, action_type):
                if i > 15:
                    return None, None
                action = np.random.randint(5 * 3)  # random choice of 3 actions on 3 darts
                dart_id = dart_indices[int(action / 3)]
                action_type = action % 3
                i += 1
        else:
            X, dart_indices = self.env.get_x(state, None)
            X = torch.tensor(X, dtype=torch.float32)
            pmf = self.forward(X)
            dist = Categorical(pmf)
            action = dist.sample()
            action = action.tolist()
            prob = pmf[action]
            action_darts = int(action/3)
            action_type = action % 3
            dart_id = dart_indices[action_darts]
            i = 0
            while not isValidAction(state, dart_id, action_type):
                if i > 15:
                    return None, None
                pmf = self.forward(X)
                dist = Categorical(pmf)
                action = dist.sample()
                action = action.tolist()
                prob = pmf[action]
                action_darts = int(action/3)
                action_type = action % 3
                dart_id = dart_indices[action_darts]
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

    def get_pi(self, state):
        X, _ = self.env.get_x(state, None)
        X = torch.tensor(X, dtype=torch.float32)
        pmf = self.forward(X)
        return pmf.tolist()

    def update(self, delta, L, state, action):
        X, _ = self.env.get_x(state, None)
        X = torch.tensor(X, dtype=torch.float32)
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