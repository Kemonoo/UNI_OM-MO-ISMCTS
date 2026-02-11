import random
import math

from phantom_ttt_state import PhantomTTTState

class Node():   #v
    def __init__(self, action=None, parent=None):
        # Structure
        self.children = []      # c(v) List of child nodes
        self.action = action    # a(v) Move taken to get to this node
        self.parent = parent    # Parent node

        # Stats
        self.visits = 0                 # n(v)
        self.availability_count = 0     # n'(v)
        self.total_reward = 0.0         # r(v)

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

    def get_action(self, state):
        # Initialize
        root = Node()

        for _ in range(self.iterations):
            # Initialize
            current_node = root
            visited_nodes = []  # Visited nodes with available compatible children

            # # DEBUG
            # if _ % 100 == 0:
            #     print(f"________ iterations: {_} ________")
            #     for child in root.children:
            #         win_rate = child.total_reward / child.visits if child.visits > 0 else 0
            #         print(f"{child.action:<10} | {child.visits:<12} | {child.availability_count:<12} | {child.total_reward:<10.1f} | {win_rate:<10.2f}")

            # A: Determinization
            d = state.determinize()

            # B: Selection
            while not d.is_terminal() and len(self._get_untried_actions(d, current_node)) == 0:
                compatible_children = self._get_compatible_children(d, current_node)

                # Select child with highest UCB
                best_child = max(compatible_children, key=lambda c: c._ucb())

                # Update
                visited_nodes.append((best_child, compatible_children))
                d.apply_action(best_child.action)
                current_node = best_child

            # C: Expansion
            if not d.is_terminal():
                action = random.choice(self._get_untried_actions(d, current_node))
                new_child = Node(action=action, parent=current_node)
                current_node.children.append(new_child)

                # Update
                compatible_children = self._get_compatible_children(d, current_node)
                visited_nodes.append((new_child, compatible_children))
                d.apply_action(action)
                current_node = new_child

            # D: Simulation
            while not d.is_terminal():
                action = random.choice(d.get_true_state_actions())
                d.apply_action(action)

            reward = d.get_reward(self.player)

            # E: Backpropagation
            for node, available_nodes in visited_nodes:
                node.visits += 1
                node.total_reward += reward

                for n in available_nodes:
                    n.availability_count += 1

            # also update the root
            root.visits += 1
            root.total_reward += reward

        # Return best action
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
            

    def _get_compatible_children(self, d: PhantomTTTState, node: Node):
        legal_actions_set = set(d.get_legal_actions())
        compatible_children = [
            c for c in node.children
            if c.action in legal_actions_set
        ]
        return compatible_children


    def _get_untried_actions(self, d: PhantomTTTState, node: Node):
        # Returns untried action compatible with current determinization and the current tree status
        legal_moves = d.get_legal_actions()
        existing_actions = {child.action for child in node.children}
        return [action for action in legal_moves if action not in existing_actions]
