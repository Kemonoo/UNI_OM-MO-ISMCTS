
class PhantomTTTState():
    def __init__(self):
        # 0 - empty, 1 - P1, 2 - P2
        self.board = [0] * 9

        self.current_player = 1     # P1 -> 1; P2 -> 2
        

    def apply_action(self, action):
        pass

    
    # ISMCTS Functions
    def determinise(self, observer_id):
        pass

    def get_observation(self, viewer_id, action):
        pass

    def get_legal_actions(self, player):
        pass
