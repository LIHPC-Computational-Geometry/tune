import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from mesh_model.mesh_analysis_old import isValidAction


class NaNException(Exception):
    pass


class NNPolicy(nn.Module):
    def __init__(self, env, input_dim, hidden_dim, output_dim, gamma, lr=0.0001):
        super(NNPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gamma
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.env = env

    def reset(self, env=None):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.optimizer = Adam(self.parameters(), lr=self.optimizer.defaults['lr'],
                              weight_decay=self.optimizer.defaults['weight_decay'])

    def select_action(self, state):
        if np.random.rand() <= 0.01:
            X, dart_indices = self.env.get_x(state, None)
            action = np.random.randint(5)
            dart_id = dart_indices[action]
        else:
            X, dart_indices = self.env.get_x(state, None)
            X = torch.tensor(X, dtype=torch.float32)
            pmf = self.forward(X)
            dist = Categorical(pmf)
            action = dist.sample()
            action = action.tolist()
            dart_id = dart_indices[action]
            i = 0
            while not isValidAction(state, dart_id) and i < 10:
                pmf = self.forward(X)
                dist = Categorical(pmf)
                action = dist.sample()
                action = action.tolist()
                dart_id = dart_indices[action]
                i += 1
        return action, dart_id

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("X contient des valeurs nan ou infinies")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise NaNException("La fonction softmax renvoie nan ou infinies")
        return x

    def get_pi(self, state):
        X, _ = self.env.get_x(state, None)
        X = torch.tensor(X, dtype=torch.float32)
        pmf = self.forward(X)
        return pmf.tolist()

    def update(self, trajectory):
        G = 0
        policy_loss = []
        for _, (s, a, r) in enumerate(reversed(trajectory), 1):
            G = r + self.gamma * G
            X, _ = self.env.get_x(s, None)
            X = torch.tensor(X, dtype=torch.float32)
            action = torch.tensor(a[0], dtype=torch.int64)
            pmf = self.forward(X)
            log_prob = torch.log(pmf[action])
            policy_loss.append(-log_prob * G)
        self.learn(policy_loss)

    def learn(self, policy_loss):
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()


