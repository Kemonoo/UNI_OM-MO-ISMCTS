from phantom_ttt_state import PhantomTTTState

from agents.human import HumanPlayer
from agents.random import RandomAgent


game = PhantomTTTState()
a1 = HumanPlayer(1)
a2 = RandomAgent(2)

while not game.is_terminal():
    current_player = game.current_player

    if current_player == 1:
        action = a1.get_action(game)
    else:
        action = a2.get_action(game)

    game.apply_action(action)

    print("_______________________\n")
    print("Current true board")
    print(game)
    print("_______________________")

print(game.winner)

def play_game(p1, p2):
    pass