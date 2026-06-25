# OM-MO-ISMCTS in Phantom Tic-Tac-Toe

Code for my bachelor thesis. It implements MO-ISMCTS and an extended version
(OM-MO-ISMCTS) that adds Bayesian opponent modelling, and runs them in Phantom
Tic-Tac-Toe.

Phantom Tic-Tac-Toe is 3x3 tic-tac-toe where you cannot see the opponent's
marks. You only find out a square is taken by trying to play it and colliding,
which costs you the turn but reveals that square.

The OM agent keeps a belief over a small set of opponent types and uses it in two
places: when filling in the hidden pieces (determinization), and when choosing
opponent moves inside the search tree (selection). The belief is updated after
each of the agent's own moves, based on whether the move was a success or a
collision.

## Agents

There are four configurations, all tested against the same three opponents:

- Baseline: plain MO-ISMCTS, no opponent model (agents/mo_ismcts.py)
- OM-Det: belief-biased determinization only
- OM-Sel: belief-biased opponent selection only
- OM-Full: both of the above (the proposed agent)

The OM agents are in agents/om_mo_ismcts.py and are picked with the `mode`
argument. The opponents (agents/heuristics.py) are RandomAgent, CenterAgent and
CornerAgent. The belief vector is ordered [Random, Center, Corner] everywhere.

## Setup

Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

Run the scripts in this order. The settings (number of games, iteration budgets,
seed) are constants near the bottom of each file.

```
python test_iterations.py           # preliminary iteration sweep (Figure 4.1)
python benchmark_timing.py          # timing benchmark, single process
python experiment.py                # main ablation, this is the long one

python make_graphs.py [json]        # makes the figures in data/
python print_all_metrics.py [json]  # prints all win rates / times / beliefs
python analyze_results.py json      # z-tests and claim checks
python belief_argmax.py json        # RQ3 argmax rates
```

make_graphs.py and print_all_metrics.py use the newest result file in data/ if
you do not pass a path.

Note on timing: experiment.py runs games in parallel, so the move times it
records are too high because of CPU contention. The timing numbers used in the
thesis come from benchmark_timing.py, which runs one process at a time.

## Parameters

- Main ablation: 10,000 games per seat, so 20,000 per agent/opponent/budget
  (4 agents x 3 opponents x 2 budgets = 480,000 games in total)
- Iteration budgets: 500 and 2000
- Preliminary sweep: budgets from 100 to 10,000
- Exploration constant C = 0.7
- Seed: 42 (each game uses seed + i)

The same settings are also stored in the config block of each result JSON.

## Output

Everything is written to data/. The result JSON files are git-ignored, because
the main one is around 400 MB and can just be regenerated. The figures
(data/*.png) are kept in the repo. The runs are seeded, so the baseline
reproduces to about 3 decimals.

## Files

```
phantom_ttt_state.py   game state and determinization
agents/
  heuristics.py        the three opponents
  mo_ismcts.py         MO-ISMCTS baseline
  om_mo_ismcts.py      OM-MO-ISMCTS and the belief update
experiment.py          main ablation
test_iterations.py     preliminary iteration sweep
benchmark_timing.py    timing benchmark
make_graphs.py         figures
print_all_metrics.py   text metrics
analyze_results.py     z-tests and claim checks
belief_argmax.py       RQ3 argmax metric
naming.py              agent name mapping used by the scripts
```
