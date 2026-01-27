

class Node():
    def __init__(self, move=None, parent=None, state=None):
        self.move = move    # Move taken to get to this node
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        
        # Tracks how many times was this node available
        self.availability_count = 0


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