from phantom_ttt_state import PhantomTTTState

from agents.human import HumanPlayer
from agents.random import RandomAgent

from agents.so_ismcts import SOISMCTS


# game = PhantomTTTState()
# # a1 = HumanPlayer(1)
# a1 = SOISMCTS(1, 10000)
# a2 = RandomAgent(2)

# while not game.is_terminal():
#     current_player = game.current_player

#     if current_player == 1:
#         action = a1.get_action(game)
#     else:
#         action = a2.get_action(game)

#     game.apply_action(action)

#     print("_______________________\n")
#     print("Current true board")
#     print(game)
#     print("_______________________")

# print("_______________________")
# print("Game over")
# print(game)
# print(f"Player {game.winner} won.")
# print("_______________________")

def play_game(p1=RandomAgent(1), p2=RandomAgent(2)):
    game = PhantomTTTState()

    while not game.is_terminal():
        current_player = game.current_player

        if current_player == 1:
            action = p1.get_action(game)
        else:
            action = p2.get_action(game)

        game.apply_action(action)

    return game.winner


def play_n_games(p1, p2, n):
    score = [0, 0, 0]
    for game in range(n):
        if game % 10 == 0:
            print(f"{game} games done.")
        score[play_game(p1, p2)] += 1

    print(f"P1 is {p1.__class__.__name__}; P2 is {p2.__class__.__name__}")
    print(f"P1 won: {score[1]}; P2 won: {score[2]}; Draws: {score[0]}")


if __name__ == '__main__':
    p1 = SOISMCTS(1, 1000)
    p2 = RandomAgent(2)

    play_game(p1, p2)
    # play_n_games(p1, p2, 1000)