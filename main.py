import sys

from user_game import user_game
from train import train


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if len(sys.argv) == 2:
        user_game(int(sys.argv[1]))
    else:
        train()
