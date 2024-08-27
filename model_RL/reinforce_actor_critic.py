import torch
from tqdm import tqdm
from model_RL.utilities.actor_critic_networks import NaNExceptionActor, NaNExceptionCritic


def reinforce_actor_critic(actor, critic, env, nb_epochs=1, nb_episodes=100, batch_size=5):
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
    nb_samples = 0
    try:
        for _ in tqdm(range(nb_epochs)):
            rollouts = []
            for _ in tqdm(range(nb_episodes)):
                env.reset()
                trajectory = []
                ep_reward = 0
                I = 1
                batch = []
                nb_updates = 0
                done = False

                while True:
                    state = env.mesh
                    action = actor.select_action(state)
                    env.step(action)
                    next_state = env.mesh
                    R = env.reward
                    ep_reward += env.reward
                    trajectory.append((state, action, R, next_state, I, done))
                    batch.append((state, action, R, next_state, I, done))
                    nb_samples += 1
                    I = 0.9 * I
                    if env.terminal:
                        if env.won:
                            wins.append(1)
                            done = True
                        else:
                            wins.append(0)
                        break
                    if nb_samples >= batch_size:
                        critic_loss = []
                        actor_loss = []

                        critic.optimizer.zero_grad()
                        for _, (s, a, r, next_s, I, done) in enumerate(batch, 1):
                            X, indices_faces = env.get_x(s, None)
                            X = torch.tensor(X, dtype=torch.float32)
                            next_X, _ = env.get_x(next_s, None)
                            next_X = torch.tensor(next_X, dtype=torch.float32)
                            value = critic(X)
                            pmf = actor.forward(X)
                            log_prob = torch.log(pmf[a[0]])
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

                        batch = []
                        nb_samples = 0
                        nb_updates += 1

                rewards.append(ep_reward)
                rollouts.append(trajectory)
                len_ep.append(len(trajectory))

    except NaNExceptionActor as e:
        print("NaN Exception on Actor Network")
        return None, None
    except NaNExceptionCritic as e:
        print("NaN Exception on Critic Network")
        return None, None
    return rewards, actor, wins, len_ep
