import itertools
import time

from phantom_ttt_state import PhantomTTTState
from agents.random import RandomAgent
from agents.so_ismcts import SOISMCTS
from agents.mo_ismcts import MOISMCTS

# ==========================================
# TOURNAMENT CONFIGURATION
# ==========================================
GAMES_PER_MATCHUP = 500
MCTS_ITERATIONS = 1000

# Define the agents. Using lambdas allows us to initialize them 
# with the correct player_id (1 or 2) right before a match begins.
AGENT_REGISTRY = {
    "Random": lambda pid: RandomAgent(pid),
    "SO_ISMCTS": lambda pid: SOISMCTS(pid, MCTS_ITERATIONS),
    "MO_ISMCTS": lambda pid: MOISMCTS(pid, MCTS_ITERATIONS)
}

def play_matchup(name_p1, name_p2, games_to_play):
    """Runs a set of games between two specific agents and returns the stats."""
    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    print(f"\nMatchup: {name_p1} (P1) vs {name_p2} (P2)")
    
    for g in range(games_to_play):
        # 1. Print progress bar using carriage return (\r) to overwrite the same line
        pct = int(((g + 1) / games_to_play) * 100)
        progress_bar = "#" * (pct // 5) + "-" * (20 - (pct // 5))
        print(f"\r  [{progress_bar}] {g + 1}/{games_to_play} ({pct}%) played...", end="", flush=True)

        # 2. Instantiate fresh agents for the game (resets any internal state)
        p1 = AGENT_REGISTRY[name_p1](1)
        p2 = AGENT_REGISTRY[name_p2](2)

        # 3. Play the game
        game = PhantomTTTState()
        while not game.is_terminal():
            if game.current_player == 1:
                move = p1.get_move(game)
            else:
                move = p2.get_move(game)
            game.apply_move(move)

        # 4. Record results
        if game.winner == 1:
            wins_p1 += 1
        elif game.winner == 2:
            wins_p2 += 1
        else:
            draws += 1

    print() # Clear the line after the progress bar finishes
    return wins_p1, wins_p2, draws

def print_results_table(results):
    """Prints the tournament results in a nicely formatted ASCII table."""
    print("\n" + "="*105)
    print(f"{'TOURNAMENT RESULTS':^105}")
    print("="*105)
    
    header = f"{'P1 (Player 1)':<15} | {'P2 (Player 2)':<15} | {'P1 Wins':<9} | {'P2 Wins':<9} | {'Draws':<7} | {'P1 Win%':<9} | {'P2 Win%':<9} | {'Draw%':<7}"
    print(header)
    print("-" * 105)
    
    for r in results:
        row = f"{r['P1']:<15} | {r['P2']:<15} | {r['P1_Wins']:<9} | {r['P2_Wins']:<9} | {r['Draws']:<7} | {r['P1_WinRate']:>8.1f}% | {r['P2_WinRate']:>8.1f}% | {r['DrawRate']:>6.1f}%"
        print(row)
        
    print("="*105)

def run_tournament():
    agent_names = list(AGENT_REGISTRY.keys())
    
    # itertools.product generates all possible combinations, including self-matchups.
    # e.g., (A, A), (A, B), (A, C), (B, A), (B, B), etc.
    matchups = list(itertools.product(agent_names, repeat=2))
    
    print(f"Starting Tournament: {len(agent_names)} agents, {len(matchups)} total matchups.")
    print(f"Games per matchup: {GAMES_PER_MATCHUP} | MCTS Iterations: {MCTS_ITERATIONS}")
    
    results = []
    start_time = time.time()
    
    for p1_name, p2_name in matchups:
        wins_p1, wins_p2, draws = play_matchup(p1_name, p2_name, GAMES_PER_MATCHUP)
        
        # Calculate percentages
        p1_winrate = (wins_p1 / GAMES_PER_MATCHUP) * 100
        p2_winrate = (wins_p2 / GAMES_PER_MATCHUP) * 100
        draw_rate = (draws / GAMES_PER_MATCHUP) * 100
        
        results.append({
            'P1': p1_name,
            'P2': p2_name,
            'P1_Wins': wins_p1,
            'P2_Wins': wins_p2,
            'Draws': draws,
            'P1_WinRate': p1_winrate,
            'P2_WinRate': p2_winrate,
            'DrawRate': draw_rate
        })
        
    print(f"\nTournament finished in {time.time() - start_time:.1f} seconds.")
    print_results_table(results)

if __name__ == '__main__':
    run_tournament()