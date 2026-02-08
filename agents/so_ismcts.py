import math
import random

from phantom_ttt_state import PhantomTTTState

class Node():   #v
    def __init__(self, action=None, parent=None, state=None):
        self.state = state 

        # Structure
        self.children = []      # c(v) List of child nodes
        self.action = action    # a(v) Move taken to get to this node
        self.parent = parent    # Parent node

        # Stats
        self.visits = 0                 # n(v)
        self.availability_count = 0     # n'(v)
        self.total_reward = 0.0                 # r(v)

    def _ucb(self, exploration_constant=0.7):

        if self.visits == 0 or self.availability_count == 0:
            return float('inf')
        
        # Exploitation
        exploitation = self.total_reward / self.visits

        # Exploration
        exploration = exploration_constant * math.sqrt((math.log(self.availability_count) / self.visits) )

        return exploitation + exploration

class SOISMCTS():
    def __init__(self, player, iterations):
        self.player = player
        self.iterations = iterations

    def choose_action(self, state):
        # Initialize
        root = Node(state=state)

        for _ in range(self.iterations):
            # Initialize
            current_node = root

            # A: Determinization
            d = state.determinize()

            # B: Selection
            while not d.is_terminal() and len(self._get_untried_actions(d, current_node)) == 0:
                
                # Get compatible children
                legal_actions_set = set(d.get_legal_actions())
                
                compatible_children = [
                    c for c in current_node.children 
                    if c.action in legal_actions_set
                    ]

                # Select child with highest UCB
                best_child = max(compatible_children, key=lambda c: c._ucb())

                d.apply_action(best_child.action)
                current_node = best_child

            # C: Expansion
            if not d.is_terminal():
                action = random.choice(self._get_untried_actions(d, current_node))
                new_child = Node(action=action, parent=current_node)
                current_node.children.append(new_child)

                d.play_action(action)
                current_node = new_child

            # D: Simulation
            while not d.is_terminal():
                action = random.choice(d.get_legal_actions())
                d.apply_action(action)

            reward = d.get_reward(self.player)

            # E: Backpropagation
            
            

    def _get_untried_actions(self, d: PhantomTTTState, node: Node):
        # Returns untried action compatible with current determinization and the current tree status
        legal_moves = d.get_legal_actions()
        existing_actions = {child.action for child in node.children}
        return [action for action in legal_moves if action not in existing_actions]
