import numpy as np
import torch
import torch.nn as nn
from torch.distributed.argparse_util import env
from torch.optim import Adam
from torch.distributions import Categorical

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


class NaNExceptionActor(Exception):
    pass


class NaNExceptionCritic(Exception):
    pass


class Actor(nn.Module):
    def __init__(self, env, input_dim, output_dim, lr=0.0001):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = 0.9
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.env = env

    def reset(self, env=None):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.optimizer = Adam(self.parameters(), lr=self.optimizer.defaults['lr'], weight_decay=self.optimizer.defaults['weight_decay'])

    def select_action(self, state):
        if np.random.rand() <= 0.1:
            X, indices_faces = self.env.get_x(state, None)
            action = np.random.randint(20)
            face_id = int(action / 4)
            face_id = indices_faces[face_id]
            action = [action % 4, face_id, action]
        else:
            X, indices_faces = self.env.get_x(state, None)
            X = torch.tensor(X, dtype=torch.float32)
            pmf = self.forward(X)
            dist = Categorical(pmf)
            action = dist.sample()
            action = action.tolist()
            face_id = int(action/4)
            face_id = indices_faces[face_id]
            action = [action % 4, face_id, action]
        return action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise NaNExceptionActor("Les couches cachées renvoient nan ou infinies")
        return self.softmax(x)

    def get_pi(self, state):
        X, indices_faces = self.env.get_x(state, None)
        X = torch.tensor(X, dtype=torch.float32)
        pmf = self.forward(X)
        return pmf.tolist()

    def update(self, delta, I, state, action):
        X, indices_faces = self.env.get_x(state, None)
        X = torch.tensor(X, dtype=torch.float32)
        action = torch.tensor(action[2], dtype=torch.int64)
        pmf = self.forward(X)
        log_prob = torch.log(pmf[action])
        actor_loss = -log_prob * delta * I
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

    def learn(self, critic_loss ):
        self.optimizer.zero_grad()
        critic_loss = torch.stack(critic_loss).sum()
        critic_loss.backward()
        self.optimizer.step()


