import random

from phantom_ttt_state import PhantomTTTState
class RandomAgent():
    def __init__(self, player_id: int):
        self.player_id = player_id 

    def get_action(self, state: PhantomTTTState):
        valid_moves = state.get_legal_actions(state.current_player)

        if not valid_moves:
            raise ValueError("no valid moves")
        
        return random.choice(valid_moves)
        