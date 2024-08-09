import torch
from tqdm import tqdm
from model_RL.utilities.actor_critic_networks import NaNExceptionActor, NaNExceptionCritic, Actor, Critic


class AC:
    def __init__(self, env, lr, gamma, nb_episodes):
        """
        Initialize the actor and critic networks. Monte-Carlo actor-critic algorithm.
        Learning takes place at the end of each episode over the entire trajectory.
        Similar to A2C (Advantage Actor-Critic algorithm)
        :param env: the environment
        :param lr: learning rate
        :param gamma: discount factor
        :param nb_episodes: number of episodes to train for
        """
        self.env = env
        self.actor = Actor(env, 30, 5, lr=0.0001)
        self.critic = Critic(30, lr=0.0001)
        self.lr = lr
        self.gamma = gamma
        self.nb_episodes = nb_episodes

    def train(self) -> [Actor, list, list, list]:
        """
        Train the model over nb episodes. Both Actor and Critic networks are updated at the end of each episode.
        :return: the final actor policy, rewards history, wins history and number of steps history
        """
        rewards = []
        wins = []
        len_ep = []
        try:
            for ep in tqdm(range(self.nb_episodes)):
                self.env.reset()
                trajectory = []
                ep_reward = 0
                I = 1
                actor_loss = []
                critic_loss = []

                while True:
                    state = self.env.mesh
                    action = self.actor.select_action(state)
                    self.env.step(action)
                    next_state = self.env.mesh
                    R = self.env.reward
                    ep_reward += self.env.reward
                    trajectory.append((state, action, self.env.reward))
                    # Actor Critic update
                    X, indices_faces = self.env.get_x(state, None)
                    X = torch.tensor(X, dtype=torch.float32)
                    next_X, next_indices_faces = self.env.get_x(next_state, None)
                    next_X = torch.tensor(next_X, dtype=torch.float32)
                    value = self.critic(X)
                    next_value = 0 if self.env.terminal else self.critic(next_X)
                    delta = R + 0.9 * next_value - value
                    critic_loss.append(self.critic.update(delta, value))
                    actor_loss.append(self.actor.update(delta.detach(), I, state, action))
                    I = 0.9 * I
                    if self.env.terminal:
                        len_ep.append(len(trajectory))
                        if self.env.won:
                            wins.append(1)
                        else:
                            wins.append(0)
                        break
                self.critic.learn(critic_loss)
                self.actor.learn(actor_loss)
                rewards.append(ep_reward)
        except NaNExceptionActor as e:
            print("NaN Exception on Actor Network")
            return None, None
        except NaNExceptionCritic as e:
            print("NaN Exception on Critic Network")
            return None, None
        return self.actor, rewards, wins, len_ep
