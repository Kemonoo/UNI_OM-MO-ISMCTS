from phantom_ttt_state import PhantomTTTState
class HumanPlayer():
    def __init__(self, player_id: int):
        self.player_id = player_id

    def get_action(self, state: PhantomTTTState):
        valid_moves = state.get_legal_actions_for_current_player()

        if not valid_moves:
            raise ValueError("no valid moves")
        
        action = self._display_state(state)
        while action not in valid_moves:
            print("That's an invalid action")
            print(f"Choose one of these actions {valid_moves}")
            action = self._display_state(state)
        return action


    def _display_state(self, state):
        print(f"You are player: {self.player_id}")
        print("This is your board\n\n")

        print(state.to_string_for_player(state.current_player))
        # for i in range(3):
        #     print(f"| {obs[i*3]} | {obs[i*3] + 1} | {obs[i*3] + 2} |")

        try:
            return int(input("\nChoose valid action (0-8): "))
        except ValueError:
            return -1
        
        