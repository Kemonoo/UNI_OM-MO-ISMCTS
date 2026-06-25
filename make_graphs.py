import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

import naming

# Experiment configuration and the visualization palette. Agent naming
# (stored key <-> OM-X display label, colors) lives in naming.py.
OPPONENTS = ["Random", "Center-Heavy", "Corner-Heavy"]
AGENTS = naming.DISPLAY_AGENTS          # display order: Baseline, OM-Det, OM-Sel, OM-Full
AGENT_STORED = naming.STORED            # display name -> stored name (for reading the JSON)
AGENT_COLORS = naming.COLORS

def load_data(filepath):
    """Loads the results JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_win_rates(data, xmin=75, xmax=93):
    """Plots overall win rates as horizontal grouped bars, one panel per budget.

    Horizontal bars keep the value labels at the bar tips (instead of colliding
    above vertical bars), and the win-rate axis is truncated to [xmin, xmax] so
    the differences between agents (all clustered in ~80-91%) are visible. Sized
    for a full-width figure --- include it in the thesis with `figure*` /
    \\textwidth, since 24 labelled bars do not fit a single column.
    """
    results = data['results']
    iteration_budgets = data['config']['iteration_budgets']

    fig, axes = plt.subplots(len(iteration_budgets), 1, figsize=(7.0, 4.0 * len(iteration_budgets)),
                             sharex=True)
    if len(iteration_budgets) == 1: axes = [axes]

    fig.suptitle('Overall Win Rates by Iteration Budget', fontsize=15, fontweight='bold')

    y = np.arange(len(OPPONENTS))
    bar_h = 0.92 / len(AGENTS)

    for i, iters in enumerate(iteration_budgets):
        ax = axes[i]
        ax.set_title(f'{iters} Iterations', fontsize=13, pad=6)

        # Aggregate win rates (keyed by the stored agent name in the JSON)
        agent_data = defaultdict(lambda: defaultdict(list))
        for r in results:
            if r['iterations'] == iters:
                winrate = r['p1_winrate'] if r['agent_first'] else r['p2_winrate']
                agent_data[r['agent']][r['opponent']].append(winrate)

        # One horizontal bar per agent, grouped by opponent (top agent first)
        for j, agent in enumerate(AGENTS):
            stored = AGENT_STORED[agent]
            winrates = [np.mean(agent_data[stored][opp]) if agent_data[stored][opp] else 0 for opp in OPPONENTS]
            offset = (j - (len(AGENTS) - 1) / 2) * bar_h

            bars = ax.barh(y + offset, winrates, bar_h, label=agent,
                           color=AGENT_COLORS[agent], alpha=0.9, zorder=3)

            for bar in bars:
                w = bar.get_width()
                if w > 0:
                    ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2.,
                            f'{w:.2f}%', va='center', ha='left', fontsize=8.5)

        ax.set_yticks(y)
        ax.set_yticklabels(OPPONENTS, fontsize=11)
        ax.invert_yaxis()                     # first opponent at the top
        ax.set_xlim(xmin, xmax)
        ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
        if i == len(iteration_budgets) - 1:
            ax.set_xlabel('Win Rate (%)', fontsize=12, labelpad=8)

    # Shared legend across the top of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=len(AGENTS), frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs('data', exist_ok=True)
    save_path = 'data/win_rates.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

def plot_timing(data, timing_data=None):
    """Plots the average move time per agent per budget.

    Preferred source is `timing_data`, the single-threaded benchmark produced by
    benchmark_timing.py (means with std error bars). Those numbers are
    representative of single-agent latency. If no benchmark is provided, it falls
    back to the main run's avg_time_p1/p2 (measured under CPU contention),
    selecting only the evaluated agent's seat (the previous version averaged in
    the heuristic opponent's ~0 s seat, halving every value).
    """
    use_bench = timing_data is not None
    if use_bench:
        budgets = timing_data['config']['budgets']
        bench = timing_data['timing']
    else:
        budgets = data['config']['iteration_budgets']
        results = data['results']

    fig, axes = plt.subplots(len(budgets), 1, figsize=(8, 5 * len(budgets)))
    if len(budgets) == 1: axes = [axes]

    fig.suptitle('Average Move Time by Iteration Budget',
                 fontsize=16, fontweight='bold', y=0.97)

    x = np.arange(len(AGENTS))
    width = 0.5

    for i, iters in enumerate(budgets):
        ax = axes[i]
        ax.set_title(f'{iters} Iterations', fontsize=14, pad=10)

        means, errs = [], []
        if use_bench:
            cell = bench.get(str(iters), {})
            for agent in AGENTS:
                s = cell.get(AGENT_STORED[agent])
                means.append(s['mean'] if s else 0.0)
                errs.append(s['std'] if s else 0.0)
        else:
            # One seat per matchup is the evaluated agent; take only that seat.
            agent_times = defaultdict(list)
            for r in results:
                if r['iterations'] == iters:
                    agent_time = r['avg_time_p1'] if r['agent_first'] else r['avg_time_p2']
                    agent_times[r['agent']].append(agent_time)
            for agent in AGENTS:
                vals = agent_times[AGENT_STORED[agent]]
                means.append(np.mean(vals) if vals else 0.0)
                errs.append(np.std(vals) if vals else 0.0)

        bar_colors = [AGENT_COLORS[agent] for agent in AGENTS]
        bars = ax.bar(x, means, width, color=bar_colors, alpha=0.9, zorder=3,
                      yerr=(errs if use_bench else None), capsize=4,
                      error_kw={'ecolor': '0.3', 'lw': 1, 'zorder': 4})

        top = max([m + e for m, e in zip(means, errs)] + [1e-9])

        # Add value labels above bars
        for bar, e in zip(bars, errs):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + e + top * 0.03,
                        f'{height:.3f} s', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Subplot formatting
        ax.set_ylabel('Average Time (s)', fontsize=12, labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(AGENTS, fontsize=12)
        ax.set_ylim(0, top * 1.25)
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.3)

    os.makedirs('data', exist_ok=True)
    save_path = 'data/move_times.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def load_latest_timing_benchmark(directory='data'):
    """Return the parsed newest data/timing_benchmark_*.json, or None if absent."""
    matches = sorted(glob.glob(os.path.join(directory, 'timing_benchmark_*.json')))
    if not matches:
        return None
    print(f"Using timing benchmark: {matches[-1]}")
    return load_data(matches[-1])

def pool_belief_data(belief_analyses):
    """Calculates the weighted pooled mean and pooled standard deviation across all iteration budgets."""
    pooled = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    raw_gather = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Gather data points across all iterations
    for iters_str, iter_data in belief_analyses.items():
        for agent, opp_data in iter_data.items():
            for opp, role_data in opp_data.items():
                for role, turn_data in role_data.items():
                    for turn, metrics in turn_data.items():
                        raw_gather[agent][opp][role][turn].append(metrics)

    # Compute statistically weighted pools
    for agent, opp_data in raw_gather.items():
        for opp, role_data in opp_data.items():
            for role, turn_data in role_data.items():
                for turn, metrics_list in turn_data.items():
                    total_n = sum(m['num_samples'] for m in metrics_list)
                    if total_n == 0:
                        continue

                    # Weighted Mean
                    pooled_mean = sum(m['avg_belief_correct'] * m['num_samples'] for m in metrics_list) / total_n

                    # Pooled Variance
                    pooled_var = sum(
                        m['num_samples'] * (m['std_belief_correct']**2 + (m['avg_belief_correct'] - pooled_mean)**2)
                        for m in metrics_list
                    ) / total_n

                    pooled[agent][opp][role][turn] = {
                        'avg_belief_correct': pooled_mean,
                        'std_belief_correct': np.sqrt(pooled_var),
                        'num_samples': total_n
                    }
    return pooled

def plot_belief_convergence(data):
    """Plots belief convergence for the OM-Full agent: P1 vs P2."""
    belief_analyses = data.get('belief_analyses', {})
    if not belief_analyses:
        print("Warning: No belief data found in JSON.")
        return

    data_to_plot = pool_belief_data(belief_analyses)
    iters_used = " & ".join([str(i) for i in data['config']['iteration_budgets']])

    # 3x1 Vertical Grid
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)
    fig.suptitle(f'OM-Full Belief Convergence: Player 1 and Player 2', fontsize=16, fontweight='bold', y=0.99)

    for i, opp in enumerate(OPPONENTS):
        ax = axes[i]
        ax.set_title(f'vs {opp}', fontsize=14, pad=10)

        # Only look at the OM-Full agent (stored key "Full-OM")
        if 'Full-OM' in data_to_plot and opp in data_to_plot['Full-OM']:
            opp_data = data_to_plot['Full-OM'][opp]

            # Plot Player 1 (Solid Line)
            if 'p1' in opp_data:
                moves_p1 = sorted([int(m) for m in opp_data['p1'].keys()])[:10]
                avg_p1 = np.array([opp_data['p1'][str(m)]['avg_belief_correct'] for m in moves_p1])
                std_p1 = np.array([opp_data['p1'][str(m)]['std_belief_correct'] for m in moves_p1])

                ax.plot(moves_p1, avg_p1, label='As Player 1', linestyle='-', marker='o', color='C0', linewidth=2)
                ax.fill_between(moves_p1, np.clip(avg_p1 - std_p1, 0, 1), np.clip(avg_p1 + std_p1, 0, 1), color='C0', alpha=0.15)

            # Plot Player 2 (Dashed Line)
            if 'p2' in opp_data:
                moves_p2 = sorted([int(m) for m in opp_data['p2'].keys()])[:10]
                avg_p2 = np.array([opp_data['p2'][str(m)]['avg_belief_correct'] for m in moves_p2])
                std_p2 = np.array([opp_data['p2'][str(m)]['std_belief_correct'] for m in moves_p2])

                ax.plot(moves_p2, avg_p2, label='As Player 2', linestyle='--', marker='s', color='C3', linewidth=2)
                ax.fill_between(moves_p2, np.clip(avg_p2 - std_p2, 0, 1), np.clip(avg_p2 + std_p2, 0, 1), color='C3', alpha=0.15)

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1.05)

        # Draw a subtle line at 0.33 to show the baseline prior
        ax.axhline(y=0.333, color='gray', linestyle=':', alpha=0.8)
        ax.xaxis.get_major_locator().set_params(integer=True)

    fig.supylabel('Probability of Correct Opponent', fontsize=12, fontweight='bold')
    fig.supxlabel('Environment Step', fontsize=12, fontweight='bold', x=0.55)

    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=12)
    plt.tight_layout()

    os.makedirs('data', exist_ok=True)
    save_path = 'data/belief_convergence_clean.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved clean convergence graph to: {save_path}")
    plt.show()

def _resolve_results(argv):
    """Results JSON from argv[1], else the newest sizeable data/ run (prelim
    files are tiny, so the size filter keeps only main-ablation outputs)."""
    if len(argv) > 1:
        return argv[1]
    cands = [p for p in glob.glob('data/experiment_results_*.json') if os.path.getsize(p) > 100_000]
    if cands:
        return max(cands, key=os.path.getmtime)
    return None


if __name__ == '__main__':
    import sys

    # Pass the results JSON as argv[1]; otherwise the newest main run in data/ is used.
    filepath = _resolve_results(sys.argv)
    if not filepath:
        print("usage: python make_graphs.py <results.json>")
        raise SystemExit(1)

    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        data = load_data(filepath)

        # Prefer the single-threaded timing benchmark (benchmark_timing.py) if one
        # exists; otherwise plot_timing falls back to the contended main-run times.
        timing_data = load_latest_timing_benchmark()

        plot_win_rates(data)
        plot_timing(data, timing_data)
        plot_belief_convergence(data)
    else:
        print(f"File {filepath} not found. Please run the experiment first.")