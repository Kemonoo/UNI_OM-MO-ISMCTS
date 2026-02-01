
from phantom_ttt_state import PhantomTTTState

class Node():
    def __init__(self, move=None, parent=None, state=None):
        # Structure
        self.move = move        # Move taken to get to this node
        self.parent = parent    # Parent Node 
        self.children = []      # 

        # Stats
        self.visits = 0
        self.reward = 0
        self.availability_count = 0

        self.untried_moves = state.get_legal_actions(state.current_player)
        


class MOISMCTS():
    def __init__(self):
        pass

    def choose_action(self, state):
        # Initialize

        # A: Determinization
        # B: Selection
        # C: Expansion
        # D: Simulation
        # E: Backpropagation
        
        pass