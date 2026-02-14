import random

from phantom_ttt_state import PhantomTTTState
class RandomAgent():
    def __init__(self, player_id: int):
        self.player_id = player_id 

    def get_move(self, state: PhantomTTTState):
        valid_moves = state.get_legal_moves()

        if not valid_moves:
            raise ValueError("no valid moves")
        
        return random.choice(valid_moves)
        