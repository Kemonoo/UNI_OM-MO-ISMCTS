"""
benchmark_timing.py
===================

Single-threaded timing benchmark for the OM-MO-ISMCTS agents.

The main study (experiment.py) measures per-move wall-clock inside a
ProcessPoolExecutor with max_workers = os.cpu_count(), so on a 6-core / 12-thread
machine every reported time is measured under ~12-way CPU contention - not
representative of a single agent's latency. This benchmark re-measures timing in
a dedicated SINGLE-PROCESS pass (no contention), timing the full agent turn
(get_move + apply_move + bayesian_update) with time.perf_counter(), and records
ONLY the evaluated agent's seat (the heuristic opponent's ~0 s moves are skipped).

It reuses the exact agent/opponent factories from experiment.py so the agents are
identical to the main study. Results are written to data/timing_benchmark_<ts>.json
and consumed by make_graphs.py's plot_timing.

Run:  python benchmark_timing.py
"""

import time
import json
import os
import random

import numpy as np

from phantom_ttt_state import PhantomTTTState
# Reuse the SAME factories the main study uses (importing is safe: experiment.py's
# heavy work is guarded by __main__).
from experiment import AGENTS, OPPONENTS

# ---- configuration (tunable; high-budget cells dominate runtime) -----------
BUDGETS = [500, 2000]                                  # match experiment.py
AGENT_NAMES = list(AGENTS.keys())   # stored keys, in experiment.py order (auto-syncs with the rename)
GAMES_PER_CELL = 48     # split across 3 opponents x 2 seats = 6 combos x 8 games
SEED = 42


def _time_one_game(agent_factory, budget, opp_class, agent_is_p1, out_times):
    """Play one game single-process; append the evaluated agent's per-move
    full-turn times (seconds) to out_times. The opponent's moves are not timed."""
    if agent_is_p1:
        agent_pid = 1
        p1 = agent_factory(1, budget)
        p2 = opp_class(2)
    else:
        agent_pid = 2
        p1 = opp_class(1)
        p2 = agent_factory(2, budget)

    game = PhantomTTTState()
    while not game.is_terminal():
        cp = game.current_player
        mover = p1 if cp == 1 else p2
        legal_before = game.get_legal_moves()

        if cp == agent_pid:
            t0 = time.perf_counter()
            move = mover.get_move(game)
            success = game.apply_move(move)
            if hasattr(mover, 'bayesian_update'):
                mover.bayesian_update(legal_before, move, success)
            out_times.append(time.perf_counter() - t0)
        else:
            move = mover.get_move(game)
            success = game.apply_move(move)
            if hasattr(mover, 'bayesian_update'):
                mover.bayesian_update(legal_before, move, success)


def _aggregate(times, n_games):
    arr = np.asarray(times, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
        'n_moves': int(arr.size),
        'n_games': int(n_games),
    }


def run_benchmark(budgets=BUDGETS, agent_names=AGENT_NAMES,
                  games_per_cell=GAMES_PER_CELL, seed=SEED):
    opp_classes = list(OPPONENTS.values())
    combos = [(opp, seat) for opp in opp_classes for seat in (True, False)]
    per_combo = max(1, games_per_cell // len(combos))

    print(f"Single-threaded timing benchmark (perf_counter, full agent turn).")
    print(f"Seed={seed}  budgets={budgets}  agents={agent_names}")
    print(f"games/cell={per_combo * len(combos)} "
          f"({per_combo} per opponent-seat over {len(combos)} combos)\n")

    timing = {}
    game_counter = 0
    for budget in budgets:
        timing[str(budget)] = {}
        for agent_name in agent_names:
            factory = AGENTS[agent_name]
            times = []
            n_games = 0
            t_cell = time.perf_counter()
            for opp_class in opp_classes:
                for seat in (True, False):
                    for _ in range(per_combo):
                        gseed = seed + game_counter
                        random.seed(gseed)
                        np.random.seed(gseed)
                        _time_one_game(factory, budget, opp_class, seat, times)
                        game_counter += 1
                        n_games += 1
            stats = _aggregate(times, n_games)
            timing[str(budget)][agent_name] = stats
            print(f"  [{budget:>5} iters] {agent_name:<9} "
                  f"mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  "
                  f"p95={stats['p95']:.4f}s  (n_moves={stats['n_moves']}, "
                  f"n_games={n_games}, {time.perf_counter()-t_cell:.1f}s)")

    return timing


def save_results(timing, budgets, agent_names, games_per_cell, seed):
    os.makedirs('data', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'data/timing_benchmark_{timestamp}.json'
    data = {
        'config': {
            'budgets': budgets,
            'agents': agent_names,
            'games_per_cell': games_per_cell,
            'timer': 'perf_counter',
            'scope': 'full_turn (get_move + apply_move + bayesian_update)',
            'single_threaded': True,
            'opponents': list(OPPONENTS.keys()),
            'seed': seed,
        },
        'timing': timing,
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved timing benchmark to {filename}")
    return filename


if __name__ == '__main__':
    start = time.perf_counter()
    timing = run_benchmark()
    save_results(timing, BUDGETS, AGENT_NAMES, GAMES_PER_CELL, SEED)
    print(f"Total benchmark time: {time.perf_counter() - start:.1f}s")
