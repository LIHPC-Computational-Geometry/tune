from model_RL.utilities.actor_critic_networks import NaNExceptionActor, NaNExceptionCritic, Actor, Critic
from model.mesh_analysis import global_score
import copy
import torch
import random
from tqdm import tqdm


class PPO:
    def __init__(self, env, lr, gamma, nb_iterations, nb_episodes_per_iteration, nb_epochs, batch_size):
        self.env = env
        self.actor = Actor(env, 30, 5, lr=0.0001)
        self.critic = Critic(30, lr=0.0001)
        self.lr = lr
        self.gamma = gamma
        self.nb_iterations = nb_iterations
        self.nb_episodes_per_iteration = nb_episodes_per_iteration
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.epsilon = 0.2

    def train_epoch(self, dataset):
        num_samples = len(dataset)
        print('Training on {}'.format(num_samples))
        for epoch in range(self.nb_epochs):
            start = 0
            dataset_rd = random.sample(dataset, num_samples)
            while start < num_samples - 2:
                stop = min(num_samples, start + self.batch_size)
                batch = dataset_rd[start:stop]
                critic_loss = []
                actor_loss = []
                self.critic.optimizer.zero_grad()
                G = 0
                for i, (s, a, r, old_prob, next_s, done) in enumerate(batch, 1):
                    X, indices_faces = self.env.get_x(s, None)
                    X = torch.tensor(X, dtype=torch.float32)
                    next_X, next_indices_faces = self.env.get_x(next_s, None)
                    next_X = torch.tensor(next_X, dtype=torch.float32)
                    value = self.critic(X)
                    pmf = self.actor.forward(X)
                    log_prob = torch.log(pmf[a[0]])
                    next_value = torch.tensor(0.0, dtype=torch.float32) if done else self.critic(next_X)
                    delta = r + 0.9 * next_value - value
                    G = (r + 0.9 * G) / 10
                    st = global_score(s)[1]
                    ideal_s = global_score(s)[2]
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

    def train(self):
        """
        Train the PPO model
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
                for ep in range(self.nb_episodes_per_iteration):
                    self.env.reset()
                    trajectory = []
                    ep_reward = 0
                    done = False
                    while True:
                        state = copy.deepcopy(self.env.mesh)
                        action = self.actor.select_action(state)
                        X, dart_indices = self.env.get_x(state, None)
                        X = torch.tensor(X, dtype=torch.float32)
                        pmf = self.actor.forward(X)
                        prob = pmf[action[0]]
                        self.env.step(action)
                        next_state = copy.deepcopy(self.env.mesh)
                        R = self.env.reward
                        ep_reward += self.env.reward
                        if self.env.terminal:
                            if self.env.won:
                                wins.append(1)
                                done = True
                                trajectory.append((state, action, R, prob, next_state, done))
                            else:
                                wins.append(0)
                                trajectory.append((state, action, R, prob, next_state, done))
                            break
                        trajectory.append((state, action, R, prob, next_state, done))
                    rewards.append(ep_reward)
                    rollouts.append(trajectory)
                    dataset.extend(trajectory)
                    len_ep.append(len(trajectory))

                self.train_epoch(dataset)

        except NaNExceptionActor:
            print("NaN Exception on Actor Network")
            return None, None, None, None
        except NaNExceptionCritic:
            print("NaN Exception on Critic Network")
            return None, None, None, None
        return self.actor, rewards, wins, len_ep
