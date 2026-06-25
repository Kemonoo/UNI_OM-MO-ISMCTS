import time
import concurrent.futures
import json
import os
import random
import numpy as np

import matplotlib.pyplot as plt

from phantom_ttt_state import PhantomTTTState
from agents.mo_ismcts import MOISMCTS
from agents.om_mo_ismcts import OMMOISMCTS
from agents.heuristics import CenterAgent


def play_game(p1, p2, verbose=False):
    game = PhantomTTTState()

    while not game.is_terminal():
        current_player = game.current_player
        agent = p1 if current_player == 1 else p2
        legal_before = game.get_legal_moves()
        move = agent.get_move(game)
        success = game.apply_move(move)
        if hasattr(agent, 'bayesian_update'):
            agent.bayesian_update(legal_before, move, success)

    return game.winner


def _create_agent(agent_key, pid, iterations, mode):
    if agent_key == 'MOISMCTS':
        return MOISMCTS(pid, iterations)
    if agent_key == 'OMMOISMCTS':
        return OMMOISMCTS(pid, iterations, mode=mode)
    raise ValueError(f"Unknown agent_key {agent_key}")


def _play_single_game_process(args):
    agent_key, mode, iterations, p1_is_agent, verbose, seed = args

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if p1_is_agent:
        p1 = _create_agent(agent_key, 1, iterations, mode)
        p2 = CenterAgent(2)
    else:
        p1 = CenterAgent(1)
        p2 = _create_agent(agent_key, 2, iterations, mode)

    winner = play_game(p1, p2, verbose=verbose)
    return winner, p1_is_agent

def evaluate_agent_against_center(agent_key, mode, iterations, n_games, verbose=False, max_workers=None, seed=None):
    assert n_games % 2 == 0, "n_games must be even so half are player 1 and half player 2."
    half = n_games // 2
    agent_wins = 0
    center_wins = 0
    draws = 0

    tasks = [True] * half + [False] * half

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, p1_is_agent in enumerate(tasks):
            job_seed = None if seed is None else seed + i
            futures.append(executor.submit(_play_single_game_process, (agent_key, mode, iterations, p1_is_agent, verbose, job_seed)))

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            winner, p1_is_agent = future.result()

            if p1_is_agent:
                if winner == 1:
                    agent_wins += 1
                elif winner == 2:
                    center_wins += 1
                else:
                    draws += 1
            else:
                if winner == 2:
                    agent_wins += 1
                elif winner == 1:
                    center_wins += 1
                else:
                    draws += 1

            if verbose and (i % 50 == 0 or i == n_games):
                print(f"  Completed {i}/{n_games} games")

    total = float(n_games)
    return {
        'agent_wins': agent_wins,
        'center_wins': center_wins,
        'draws': draws,
        'agent_win_rate': agent_wins / total,
        'center_win_rate': center_wins / total,
        'draw_rate': draws / total,
    }


def run_experiment(iteration_steps, n_games, verbose=False, max_workers=None, seed=None):
    results = {
        'MOISMCTS': [],
        'OMMOISMCTS_full': []
    }

    for iterations in iteration_steps:
        print(f"Testing iterations={iterations} ...")

        mo_stats = evaluate_agent_against_center('MOISMCTS', None, iterations, n_games, verbose=verbose, max_workers=max_workers, seed=seed)
        print(f"  MOISMCTS @ {iterations}: agent_win_rate={mo_stats['agent_win_rate']:.3f}, center_win_rate={mo_stats['center_win_rate']:.3f}, draw_rate={mo_stats['draw_rate']:.3f}")

        om_stats = evaluate_agent_against_center('OMMOISMCTS', 'full', iterations, n_games, verbose=verbose, max_workers=max_workers, seed=seed)
        print(f"  OMMOISMCTS(full) @ {iterations}: agent_win_rate={om_stats['agent_win_rate']:.3f}, center_win_rate={om_stats['center_win_rate']:.3f}, draw_rate={om_stats['draw_rate']:.3f}")

        results['MOISMCTS'].append((iterations, mo_stats))
        results['OMMOISMCTS_full'].append((iterations, om_stats))

    return results


def save_results(results, iteration_steps, n_games, seed=None):
    os.makedirs('data', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'data/experiment_results_{timestamp}.json'

    data = {
        'results': results,
        'config': {
            'iteration_steps': iteration_steps,
            'n_games': n_games,
            'seed': seed
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved results to {filename}")
    return filename


def plot_results(results, output_file='data/iterations_win_rates.png'):
    os.makedirs('data', exist_ok=True)
    iteration_steps = [x[0] for x in results['MOISMCTS']]
    mo_rates = [x[1]['agent_win_rate'] for x in results['MOISMCTS']]
    om_rates = [x[1]['agent_win_rate'] for x in results['OMMOISMCTS_full']]

    x = np.arange(len(iteration_steps))

    plt.figure(figsize=(10, 6))
    plt.plot(x, mo_rates, marker='o', label='MO-ISMCTS')
    plt.plot(x, om_rates, marker='o', label='OM-MO-ISMCTS (full)')
    plt.xticks(x, [str(s) for s in iteration_steps])
    plt.xlabel('Iterations')
    plt.ylabel('Win rate vs CenterAgent')
    plt.title('Iteration count vs win rate (half as P1, half as P2)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved graph to {output_file}")


if __name__ == '__main__':
    iteration_steps = [100, 200, 500, 1000, 2000, 5000, 10000]
    n_games = 3000
    seed = 42

    start = time.time()
    results = run_experiment(iteration_steps, n_games=n_games, verbose=False, seed=seed)
    duration = time.time() - start

    print(f"\nExperiment complete in {duration:.1f}s")

    plot_results(results)
    save_results(results, iteration_steps, n_games, seed=seed)

    print("\nSummary:")
    for algo in ['MOISMCTS', 'OMMOISMCTS_full']:
        print(f"\n{algo} results:")
        for iterations, stats in results[algo]:
            print(f"  {iterations:6d}: win={stats['agent_win_rate']:.3f}, center={stats['center_win_rate']:.3f}, draw={stats['draw_rate']:.3f}")