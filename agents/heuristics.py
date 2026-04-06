import random

class HeuristicAgent:
    """
    Base class for heuristic agents.
    """
    def __init__(self, player_id = None):
        self.player_id = player_id
        self.weights = None
    
    def get_move(self, state):
        valid_moves = state.get_legal_moves()

        if not valid_moves:
            raise ValueError("no valid moves")
        
        probabilities = self._get_probabilities(valid_moves)

        return random.choices(valid_moves, weights=probabilities)[0]

    def _get_probabilities(self, valid_moves):
        total_weight = 0
        for move in valid_moves:
            total_weight += self.weights[move]

        probabilities = []
        for move in valid_moves:
            probabilities.append(self.weights[move] / total_weight)

        return probabilities


class RandomAgent(HeuristicAgent):
    def __init__(self, player_id=None):
        super().__init__(player_id)
        
        self.weights = [
            1,1,1,
            1,1,1,
            1,1,1
        ]

class CenterAgent(HeuristicAgent):
    def __init__(self, player_id=None):
        super().__init__(player_id)
        
        self.weights = [
            3,1,3,
            1,10,1,
            3,1,3
        ]

class CornerAgent(HeuristicAgent):
    def __init__(self, player_id=None):
        super().__init__(player_id)
        
        self.weights = [
            10,1,10,
            1,3,1,
            10,1,10
        ]