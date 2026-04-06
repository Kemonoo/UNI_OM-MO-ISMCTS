import itertools
import time
import json
import random
import numpy as np
from collections import defaultdict
import concurrent.futures
import os

from phantom_ttt_state import PhantomTTTState
from agents.mo_ismcts import MOISMCTS
from agents.om_mo_ismcts import OMMOISMCTS
from agents.heuristics import RandomAgent, CenterAgent, CornerAgent

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
OPPONENTS = {
    "Random": RandomAgent,
    "Center-Heavy": CenterAgent,
    "Corner-Heavy": CornerAgent
}

AGENTS = {
    "Baseline": lambda pid, iters: MOISMCTS(pid, iters),
    "Det-Only": lambda pid, iters: OMMOISMCTS(pid, iters, mode='determinization_only'),
    "Sel-Only": lambda pid, iters: OMMOISMCTS(pid, iters, mode='selection_only'),
    "Full OM": lambda pid, iters: OMMOISMCTS(pid, iters, mode='full')
}

def play_game(agent_p1, agent_p2, opponent_type, track_beliefs=False):
    """Plays a single game and returns results."""
    game = PhantomTTTState()
    moves = []
    belief_history = [] if track_beliefs else None
    move_times = {1: [], 2: []}

    while not game.is_terminal():
        current_player = game.current_player
        legal_moves_before = game.get_legal_moves()
        current_agent = agent_p1 if current_player == 1 else agent_p2

        # --- TIMED PHASE START ---
        start_time = time.time()
        
        move = current_agent.get_move(game)
        success = game.apply_move(move)
        
        if hasattr(current_agent, 'bayesian_update'):
            current_agent.bayesian_update(legal_moves_before, move, success)
            
        turn_time = time.time() - start_time
        # --- TIMED PHASE END ---
        
        move_times[current_player].append(turn_time)
        moves.append((current_player, move, success))

        if track_beliefs and hasattr(current_agent, 'belief_distribution'):
            belief_history.append({
                'absolute_turn': len(moves),
                'beliefs': current_agent.belief_distribution.copy(),
                'actual_opponent': opponent_type,
                'success': success
            })

    return game.winner, move_times, belief_history 

def _worker_play_single_game(agent_name, opponent_name, iterations, agent_first, track_beliefs, seed):
    """Isolated worker to play one game."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if opponent_name == "Random": opp_class = RandomAgent
    elif opponent_name == "Center-Heavy": opp_class = CenterAgent
    elif opponent_name == "Corner-Heavy": opp_class = CornerAgent
    
    def create_om_agent(pid):
        if agent_name == "Baseline": return MOISMCTS(pid, iterations)
        elif agent_name == "Det-Only": return OMMOISMCTS(pid, iterations, mode='determinization_only')
        elif agent_name == "Sel-Only": return OMMOISMCTS(pid, iterations, mode='selection_only')
        elif agent_name == "Full OM": return OMMOISMCTS(pid, iterations, mode='full')

    if agent_first:
        p1, p2 = create_om_agent(1), opp_class(2)
    else:
        p1, p2 = opp_class(1), create_om_agent(2)

    winner, move_times, belief_history = play_game(p1, p2, opponent_name, track_beliefs)
    return winner, move_times[1], move_times[2], belief_history

def run_matchup(agent_name, opponent_name, iterations, games_to_play, agent_first=True, seed=None):
    """Runs games between an agent and an opponent using Multiprocessing."""
    if agent_first:
        p1_name, p2_name = agent_name, opponent_name
    else:
        p1_name, p2_name = opponent_name, agent_name
        
    track_beliefs = (agent_name != "Baseline")

    print(f"\nRunning: {p1_name} vs {p2_name} ({iterations} iterations)")

    wins_p1, wins_p2, draws = 0, 0, 0
    total_move_times = {1: [], 2: []}
    all_belief_histories = []

    max_workers = max(1, os.cpu_count())
    print(f"Using {max_workers} processes...")

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(games_to_play):
            job_seed = None if seed is None else seed + i
            futures.append(executor.submit(_worker_play_single_game, agent_name, opponent_name, iterations, agent_first, track_beliefs, job_seed))

        for g, future in enumerate(concurrent.futures.as_completed(futures)):
            if (g + 1) % 500 == 0 or g == 0:
                elapsed = time.time() - start_time
                print(f"  Game {g + 1}/{games_to_play} - Elapsed: {elapsed:.1f}s")

            winner, mt1, mt2, bh = future.result()

            if winner == 1: wins_p1 += 1
            elif winner == 2: wins_p2 += 1
            else: draws += 1

            total_move_times[1].extend(mt1)
            total_move_times[2].extend(mt2)
            if bh:
                all_belief_histories.extend(bh)

    p1_winrate = (wins_p1 / games_to_play) * 100
    p2_winrate = (wins_p2 / games_to_play) * 100
    draw_rate = (draws / games_to_play) * 100
    avg_time_p1 = np.mean(total_move_times[1]) if total_move_times[1] else 0
    avg_time_p2 = np.mean(total_move_times[2]) if total_move_times[2] else 0

    return {
        'p1_agent': p1_name, 'p2_agent': p2_name, 'agent': agent_name,
        'opponent': opponent_name, 'iterations': iterations, 'agent_first': agent_first,
        'wins_p1': wins_p1, 'wins_p2': wins_p2, 'draws': draws,
        'p1_winrate': p1_winrate, 'p2_winrate': p2_winrate, 'draw_rate': draw_rate,
        'avg_time_p1': avg_time_p1, 'avg_time_p2': avg_time_p2,
        'belief_histories': all_belief_histories
    }

def analyze_role_beliefs(belief_histories, opponent_name):
    """Analyze belief convergence. All passed histories belong to ONE role (P1 or P2)."""
    if not belief_histories:
        return {}

    opponent_map = {"Random": 0, "Center-Heavy": 1, "Corner-Heavy": 2}
    actual_index = opponent_map[opponent_name]

    beliefs_by_move = defaultdict(list)
    for entry in belief_histories:
        beliefs_by_move[entry['absolute_turn']].append(entry['beliefs'][actual_index])

    analysis = {}
    for move_num, beliefs in beliefs_by_move.items():
        analysis[move_num] = {
            'avg_belief_correct': np.mean(beliefs),
            'std_belief_correct': np.std(beliefs),
            'num_samples': len(beliefs)
        }
    return analysis

def save_results(results, belief_analyses, games_per_matchup, iteration_budgets, seed=None):
    os.makedirs('data', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"data/experiment_results_{timestamp}.json"

    data = {
        'results': results,
        'belief_analyses': belief_analyses,
        'config': {
            'games_per_matchup': games_per_matchup,
            'iteration_budgets': iteration_budgets,
            'opponents': list(OPPONENTS.keys()),
            'agents': list(AGENTS.keys()),
            'seed': seed
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")
    return filename

def run_experiment(games_per_matchup, iteration_budgets, seed=None):
    """Run the complete ablation study."""
    print("Starting Ablation Study for OM-MO-ISMCTS")
    
    total_matchups = len(AGENTS) * len(OPPONENTS) * len(iteration_budgets) * 2 
    print(f"Total matchups: {total_matchups}")

    results = []
    belief_analyses = {} 

    matchup_count = 0
    start_time = time.time()

    for iters in iteration_budgets:
        belief_analyses[iters] = {}
        
        for agent_name in AGENTS.keys():
            belief_analyses[iters][agent_name] = {}
            
            for opponent_name in OPPONENTS.keys():
                belief_analyses[iters][agent_name][opponent_name] = {}
                
                for agent_first in [True, False]:
                    matchup_count += 1
                    elapsed = time.time() - start_time

                    print(f"\n{'='*50}")
                    print(f"Matchup {matchup_count}/{total_matchups}")
                    print(f"Elapsed: {elapsed:.1f}s")
                    print(f"{'='*50}")

                    result = run_matchup(agent_name, opponent_name, iters, games_per_matchup, agent_first, seed=seed)
                    results.append(result)

                    if agent_name != "Baseline" and result['belief_histories']:
                        role_key = 'p1' if agent_first else 'p2'
                        analysis = analyze_role_beliefs(result['belief_histories'], opponent_name)
                        belief_analyses[iters][agent_name][opponent_name][role_key] = analysis

    filename = save_results(results, belief_analyses, games_per_matchup, iteration_budgets, seed=seed)
    print(f"\nExperiment completed! Data strictly saved to {filename}")

if __name__ == '__main__':
    games_per_matchup = 10000 
    iteration_budgets = [500, 2000]
    seed = 42

    start = time.time()
    run_experiment(games_per_matchup, iteration_budgets, seed=seed)
    duration = time.time() - start

    print(f"\nTotal experiment time: {duration:.1f}s")