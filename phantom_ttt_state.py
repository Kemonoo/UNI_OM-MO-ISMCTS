
class PhantomTTTState():
    def __init__(self):
        # 0 -> empty 
        # 1 -> P1  
        # 2 -> P2 
        self.board = [0] * 9

        self.current_player = 1     # P1 -> 1; P2 -> 2
        self.winner = None

        # Dictionary stores INDICES (0-8) a player knows are blocked by the enemy.
        self.revealed_opponent_positions = {1: set(), 2: set()}
        
    ### GAME LOGIC FUNCTIONS
    def apply_action(self, action):
        """
        Updates the board state based on the action.
        If blocked: Updates revealed info, notifies the opponent, the player does not change.
        If success: Updates board, player changes.
        """

        # Serves mainly as a bug prevention
        if action not in self.get_legal_actions():
            raise ValueError(f"Agent {self.current_player} attempted illegal move at {action}.\n")

        opponent = 3 - self.current_player

        # COLLISION
        if self.board[action] == opponent:
            self.revealed_opponent_positions[self.current_player].add(action)
            return False    # Indicate Collision

        # SUCCESS
        else:
            self.board[action] = self.current_player
            self.check_winner()

            if self.winner is None:
                self.current_player = opponent
            return True     # Indicate Success

    def get_legal_actions(self):
        """
        Returns list of indices (0 - 8) where the current player is ALLOWED to move
        based on their observation space
        """
        legal_actions = []

        known_blocked = self.revealed_opponent_positions[self.current_player]

        for i, content in enumerate(self.board):
            if content != self.current_player and i not in known_blocked:
                legal_actions.append(i)

        return legal_actions

    def check_winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in wins:
            # Check if all three squares belong to the same non-empty player
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != 0:
                self.winner = self.board[a]
                return
    
    ### ISMCTS DEPENDENT FUNCTIONS
    def get_reward(self, player_id):
        pass

    def determinise(self, observer_id):
        pass

    def get_observation(self, viewer_id, action):
        pass


    ### UTILITY FUNCTIONS
    def __str__(self):
        # print true board state
        s = ""
        for i in range(9):
            s += {0: '.', 1: 'X', 2: 'O'}[self.board[i]] + " "
            if (i + 1) % 3 == 0:
                s += "\n"
        return s
    

if __name__ == "__main__":
    state = PhantomTTTState()
    state.board = [0,0,1,0,2,2,1,0,0]
    state.revealed_opponent_positions[1].add(5)

    actions = state.get_legal_actions()
    print(actions)
