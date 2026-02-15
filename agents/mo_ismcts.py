import random
import math

from phantom_ttt_state import PhantomTTTState

class Node():   #v
    def __init__(self, move=None, parent=None):
        # Structure
        self.children = []      # c(v) List of child nodes
        self.move = move        # a(v) move = (a, v) where a ∈ {0,…,8} and v ∈ {0,1}

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

class MOISMCTS():
    def __init__(self, player, iterations):
        self.player = player
        self.iterations = iterations

    def get_move(self, state):
        # Initialize roots for both players
        roots = {1: Node(), 2: Node()}                  # Dictionary of roots

        for _ in range(self.iterations):
            # Initialize 
            current_nodes = {1: roots[1], 2: roots[2]}  # Dictionary of root nodes (that are being updated)
            visited_nodes = {1: [], 2: []}  # Visited nodes with available compatible children

            # A: Determinization
            d = state.determinize()

            # B: Selection
            while not d.is_terminal() and len(self._get_untried_moves(d, current_nodes[d.current_player])) == 0:
                compatible_children = self._get_compatible_children(d, current_nodes[d.current_player])

                # Select child with highest UCB
                best_child = max(compatible_children, key=lambda c: c._ucb())

                # Update current player
                visited_nodes[d.current_player].append((best_child, compatible_children))
                current_nodes[d.current_player] = best_child

                # Update determinization and save opponent id
                opponent = 3 - d.current_player
                success = d.apply_move(best_child.move)

                # In case 'collision' does NOT occur, the opponent's tree 
                # gets a node 'opponent moved', since no information is revealed. 
                # If 'collision' occured we do nothing for the opponent
                if success:
                    exists, child = self._find_or_create_child(d, current_nodes[opponent])
                    if not exists:
                        current_nodes[opponent].children.append(child)

                    visited_nodes[opponent].append((child, [child]))
                    current_nodes[opponent] = child


            # C: Expansion
            if not d.is_terminal():
                # Choose random move from untried moves based on observation and create a node for it
                move = random.choice(self._get_untried_moves(d, current_nodes[d.current_player]))
                new_child = Node(move=move, parent=current_nodes[d.current_player])
                current_nodes[d.current_player].children.append(new_child)


                # Update current player
                compatible_children = self._get_compatible_children(d, current_nodes[d.current_player])
                visited_nodes[d.current_player].append((new_child, compatible_children))
                current_nodes[d.current_player] = new_child

                # Update determinization and save opponent id
                opponent = 3 - d.current_player
                success = d.apply_move(new_child.move)

                # In case 'collision' does NOT occur, the opponent's tree 
                # gets a node 'opponent moved', since no information is revealed. 
                # If 'collision' occured we do nothing for the opponent
                if success:
                    exists, child = self._find_or_create_child(d, current_nodes[opponent])
                    if not exists:
                        current_nodes[opponent].children.append(child)

                    visited_nodes[opponent].append((child, [child]))
                    current_nodes[opponent] = child

            # D: Simulation
            while not d.is_terminal():
                move = random.choice(d.get_true_state_actions())
                d.apply_move(move)

            reward = d.get_reward(self.player)

            # E: Backpropagation
            updates = [
                (self.player, reward),
                (3 - self.player, -reward)
            ]

            for player, reward in updates:
                path = visited_nodes[player]
                for node, available_nodes in path:
                    node.visits += 1
                    node.total_reward += reward

                    for n in available_nodes:
                        n.availability_count += 1    
                roots[player].visits += 1
                roots[player].total_reward += reward    

        # Return best move
        best_child = max(roots[self.player].children, key=lambda c: c.visits)
        return best_child.move


    ########## HELPER FUNCTIONS ##########
    def _get_compatible_children(self, d: PhantomTTTState, node: Node):
        legal_moves_set = set(d.get_legal_moves())
        compatible_children = [
            c for c in node.children
            if c.move in legal_moves_set
        ]
        return compatible_children
    
    def _get_untried_moves(self, d: PhantomTTTState, node: Node):
        # Returns untried moves compatible with current determinization and the current tree status
        legal_moves = d.get_legal_moves()
        existing_moves = {child.move for child in node.children}
        return [move for move in legal_moves if move not in existing_moves]

    def _find_or_create_child(self, d: PhantomTTTState, node: Node):
        for child in node.children:
            if child.move == 'opponent moved':
                return True, child
        return False, Node(move = 'opponent moved', parent = node)