import random
import copy
import numpy as np

from agents.heuristics import RandomAgent, CenterAgent, CornerAgent

class PhantomTTTState():
    def __init__(self):
        # 0 -> empty 
        # 1 -> P1  
        # 2 -> P2 
        self.board = [0] * 9    # True state

        self.current_player = 1     # P1 -> 1; P2 -> 2
        self.winner = None

        # Dictionary stores INDICES (0-8) a player knows are blocked by the enemy.
        self.revealed_opponent_positions = {1: set(), 2: set()}

        self.models = [RandomAgent(), CenterAgent(), CornerAgent()]
        
    ########## GAME LOGIC FUNCTIONS ##########
    def apply_move(self, move):
        """
        Updates the board state based on the move.
        If blocked: Updates revealed info, notifies the opponent, the player does not change.
        If success: Updates board, player changes.
        """

        # Serves mainly as a bug prevention
        if move not in self.get_legal_moves():
            raise ValueError(f"Agent {self.current_player} attempted illegal move at {move}.\n")

        opponent = 3 - self.current_player

        # COLLISION
        if self.board[move] == opponent:
            self.revealed_opponent_positions[self.current_player].add(move)
            return False    # Indicate Collision

        # SUCCESS
        else:
            self.board[move] = self.current_player
            self.check_winner()

            if self.winner is None:
                self.current_player = opponent
            return True     # Indicate Success

    def get_legal_moves(self):
        """
        Returns list of indices (0 - 8) where the current player is ALLOWED to move
        based on their observation space
        """
        player_id = self.current_player

        legal_moves = []

        known_opponent_indices = self.revealed_opponent_positions[player_id]

        for i, content in enumerate(self.board):
            if content != player_id and i not in known_opponent_indices:
                legal_moves.append(i)

        return legal_moves

    def check_winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in wins:
            # Check if all three squares belong to the same non-empty player
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != 0:
                self.winner = self.board[a]
                return

        if 0 not in self.board:
            self.winner = 0
    
    def get_player_observation(self, player_id):
        """
        Return board observation of a player
        """
        player_board = list(self.board)
        known_opponent_indices = self.revealed_opponent_positions[player_id]
        opponent = 3 - player_id

        for i in range(len(player_board)):
            if player_board[i] == opponent and i not in known_opponent_indices:
                player_board[i] = 0

        return player_board


    def is_terminal(self):
        return self.winner is not None
    
    ########## ISMCTS DEPENDENT FUNCTIONS ##########
    def determinize(self, heatmap=None):
        # ---COPY---
        new_state = copy.deepcopy(self)

        # ---RESHUFFLE HIDDEN INFO---
        new_board = [0] * 9
        opponent = 3 - self.current_player

        for tile, piece in enumerate(self.board):
            if piece == self.current_player:
                new_board[tile] = piece
            elif piece == opponent and tile in self.revealed_opponent_positions[self.current_player]:
                new_board[tile] = piece

        # Find how many hidden pieces to place
        total_opponent_pieces = self.board.count(opponent)
        opponent_hidden_count = total_opponent_pieces - len(self.revealed_opponent_positions[self.current_player])

        # If there are no hidden pieces the same board is returned
        if opponent_hidden_count <= 0:
            new_state.board = new_board
            new_state.revealed_opponent_positions[opponent] = set() 
            return new_state

        # Identify candidate tiles
        player_observation = self.get_player_observation(self.current_player)
        candidates = [tile for tile, piece in enumerate(player_observation) if piece == 0]

        if heatmap is not None:
            candidate_tiles = list(heatmap.keys())
            candidate_probs = list(heatmap.values())

            selected_indices = np.random.choice(
                candidate_tiles, 
                size=opponent_hidden_count, 
                replace=False, 
                p=candidate_probs
            )
        
        else:
        # Random sampling without replacement
            random.shuffle(candidates)
            selected_indices = candidates[:opponent_hidden_count]

        for tile in selected_indices:
            new_board[tile] = opponent

        # ---UPDATE THE NEW STATE---
        new_state.board = new_board
        # We don't know what pieces has the opponent revealed
        new_state.revealed_opponent_positions[opponent] = set() 

        return new_state

    def get_reward(self, player_id):
        if self.winner == player_id: return 1.0
        if self.winner == 3 - player_id: return -1.0
        if self.winner == 0: return 0.0
        else:
            raise ValueError("There is no winner")
        
    def get_true_state_actions(self):
        # Returns list of all empty tiles
        return [action for action, piece in enumerate(self.board) if piece == 0]
        

    ########## UTILITY FUNCTIONS ##########
    def __str__(self):
        # print true board state
        s = ""
        for i in range(9):
            s += {0: '.', 1: 'X', 2: 'O'}[self.board[i]] + " "
            if (i + 1) % 3 == 0:
                s += "\n"
        return s
    
    def to_string_for_player(self, player_id):
        s = ""
        player_board = self.get_player_observation(player_id)
        for i in range(9):
            s += {0: '.', 1: 'X', 2: 'O'}[player_board[i]] + " "
            if (i + 1) % 3 == 0:
                s += "\n"
        return s
    

if __name__ == "__main__":
    pass