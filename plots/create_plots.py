import matplotlib.pyplot as plt
from numpy import ndarray


def plot_test_results(rewards: ndarray, wins: ndarray, steps: ndarray, normalized_return: ndarray) -> None:
    """
    Plot the rewards obtained on the test data, the number of times the agent wins, and the length (number of time steps)
    of each episode.
    :param rewards: list of rewards obtained on the test data
    :param wins: list of wins obtained on the test data
    :param steps: list of steps of each episode obtained on the test data
    :param normalized_return: list of normalized return obtained on the test data
    """
    nb_episodes = len(rewards)
    cat = [i for i in range(nb_episodes)]
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)  # 4 lignes, 1 colonne, graphique 1
    plt.bar(cat, rewards, label='avg_rewards')
    plt.title('Average rewards on test data')
    plt.legend()

    # Ajouter le deuxième sous-graphe
    plt.subplot(4, 1, 2)  # 4 lignes, 1 colonne, graphique 2
    plt.bar(cat, wins, label='avg_wins', color='orange')
    plt.title('Average wins on test data')
    plt.legend()

    # Ajouter le troisième sous-graphe
    plt.subplot(4, 1, 3)  # 4 lignes, 1 colonne, graphique 3
    plt.bar(cat, steps, label='avg_steps', color='green')
    plt.title('Average length of episodes on test data')
    plt.legend()

    # Ajouter le quatrième sous-graphe
    plt.subplot(4, 1, 4)  # 4 lignes, 1 colonne, graphique 4
    plt.bar(cat, normalized_return, label='avg_normalized_return', color='green')
    plt.title('Average normalized return obtained on test data')
    plt.legend()
    # Ajuster l'espacement entre les sous-graphes
    plt.tight_layout()
    # Afficher les graphiques
    plt.show()


def plot_training_results(rewards: ndarray, wins: ndarray, steps: ndarray) -> None:
    """
    Plot the rewards obtained during training, the number of times the agent wins, and the length (number of time steps)
    of each episode.
    :param rewards: list of rewards obtained during training
    :param wins: list of wins obtained during training
    :param steps: list of steps obtained during training
    """
    nb_episodes = len(rewards)
    real_runs = 1

    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Learning Rewards, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()
    plt.figure()
    plt.plot(wins)
    plt.xlabel('Episodes')
    plt.ylabel('Wins')
    plt.title('Learning Wins, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()
    plt.figure()
    plt.plot(steps)
    plt.xlabel('Episodes')
    plt.ylabel('Number of steps per episode')
    plt.title('Learning steps, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()