import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# Experiment configurations and standardized visualization palette
OPPONENTS = ["Random", "Center-Heavy", "Corner-Heavy"]
AGENTS = ["Baseline", "Det-Only", "Sel-Only", "Full OM"]

AGENT_COLORS = {
    "Baseline": "C0",
    "Det-Only": "C1",
    "Sel-Only": "C2",
    "Full OM":  "C3"
}

def load_data(filepath):
    """Loads the JSON data file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_win_rates(data):
    """Plots the overall win rates vertically for all iteration budgets."""
    results = data['results']
    iteration_budgets = data['config']['iteration_budgets']
    
    fig, axes = plt.subplots(len(iteration_budgets), 1, figsize=(10, 5 * len(iteration_budgets)))
    if len(iteration_budgets) == 1: axes = [axes]
    
    fig.suptitle('Overall Win Rates by Iteration Budget', fontsize=16, fontweight='bold', y=0.95)

    x = np.arange(len(OPPONENTS))
    width = 0.2 

    for i, iters in enumerate(iteration_budgets):
        ax = axes[i]
        ax.set_title(f'{iters} Iterations', fontsize=14, pad=10)

        # Aggregate win rates
        agent_data = defaultdict(lambda: defaultdict(list))
        for r in results:
            if r['iterations'] == iters:
                agent = r['agent']
                opponent = r['opponent']
                winrate = r['p1_winrate'] if r['agent_first'] else r['p2_winrate']
                agent_data[agent][opponent].append(winrate)

        # Plot bars per agent
        for j, agent in enumerate(AGENTS):
            winrates = [np.mean(agent_data[agent][opp]) if agent_data[agent][opp] else 0 for opp in OPPONENTS]
            offset = (j - 1.5) * width 
            
            bars = ax.bar(x + offset, winrates, width, label=agent, color=AGENT_COLORS[agent], alpha=0.9, zorder=3)
            
            # Add value labels above bars
            for bar in bars:
                height = bar.get_height()
                if height > 0: 
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Subplot formatting
        ax.set_ylabel('Win Rate (%)', fontsize=12, labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(OPPONENTS, fontsize=11)
        ax.set_ylim(50, 100)
        ax.set_yticks(np.arange(50, 101, 10))
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
        
        # Apply bottom-level formatting
        if i == len(iteration_budgets) - 1:
            ax.set_xlabel('Opponent', fontsize=12, labelpad=10)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=11)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.25)
    
    os.makedirs('data', exist_ok=True)
    save_path = 'data/win_rates.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

def plot_timing(data):
    """Plots the average execution times vertically for all iteration budgets."""
    results = data['results']
    iteration_budgets = data['config']['iteration_budgets']
    
    fig, axes = plt.subplots(len(iteration_budgets), 1, figsize=(8, 5 * len(iteration_budgets)))
    if len(iteration_budgets) == 1: axes = [axes]
    
    fig.suptitle('Average Move Time by Iteration Budget', fontsize=16, fontweight='bold', y=0.95)

    x = np.arange(len(AGENTS))
    width = 0.5 

    for i, iters in enumerate(iteration_budgets):
        ax = axes[i]
        ax.set_title(f'{iters} Iterations', fontsize=14, pad=10)

        # Aggregate move times
        agent_times = defaultdict(list)
        for r in results:
            if r['iterations'] == iters:
                agent = r['agent']
                agent_times[agent].append(r['avg_time_p1'])
                agent_times[agent].append(r['avg_time_p2'])

        times = [np.mean(agent_times[agent]) if agent_times[agent] else 0 for agent in AGENTS]
        bar_colors = [AGENT_COLORS[agent] for agent in AGENTS]

        bars = ax.bar(x, times, width, color=bar_colors, alpha=0.9, zorder=3)
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            if height > 0: 
                ax.text(bar.get_x() + bar.get_width()/2., height + (max(times) * 0.05),
                        f'{height:.3f} s', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Subplot formatting
        ax.set_ylabel('Average Time (s)', fontsize=12, labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(AGENTS, fontsize=12)
        ax.set_ylim(0, max(times) * 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.3)
    
    os.makedirs('data', exist_ok=True)
    save_path = 'data/move_times.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

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
    """Plots belief convergence for the Full OM agent: P1 vs P2."""
    belief_analyses = data.get('belief_analyses', {})
    if not belief_analyses:
        print("Warning: No belief data found in JSON.")
        return
        
    data_to_plot = pool_belief_data(belief_analyses)
    iters_used = " & ".join([str(i) for i in data['config']['iteration_budgets']])
    
    # 3x1 Vertical Grid
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)
    fig.suptitle(f'Full OM Belief Convergence: Player 1 vs Player 2', fontsize=16, fontweight='bold', y=0.99)

    for i, opp in enumerate(OPPONENTS):
        ax = axes[i]
        ax.set_title(f'vs {opp}', fontsize=14, pad=10)

        # Only look at the Full OM agent
        if 'Full OM' in data_to_plot and opp in data_to_plot['Full OM']:
            opp_data = data_to_plot['Full OM'][opp]
            
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

if __name__ == '__main__':
    # Define the target dataset generated by experiment.py
    filepath = 'data/experiment_results_20260331_163428.json'
    
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        data = load_data(filepath)
        
        plot_win_rates(data)
        plot_timing(data)
        plot_belief_convergence(data)
    else:
        print(f"File {filepath} not found. Please run the experiment first.")