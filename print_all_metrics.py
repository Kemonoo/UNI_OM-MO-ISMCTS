import json
from collections import defaultdict

def pool_belief_data(belief_analyses):
    """Mathematically pools the means across all iteration budgets."""
    pooled = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    raw_gather = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Gather data across iterations
    for iters_str, iter_data in belief_analyses.items():
        for agent, opp_data in iter_data.items():
            for opp, role_data in opp_data.items():
                for role, turn_data in role_data.items():
                    for turn, metrics in turn_data.items():
                        raw_gather[agent][opp][role][turn].append(metrics)

    # Calculate weighted pooled mean
    for agent, opp_data in raw_gather.items():
        for opp, role_data in opp_data.items():
            for role, turn_data in role_data.items():
                for turn, metrics_list in turn_data.items():
                    total_n = sum(m['num_samples'] for m in metrics_list)
                    if total_n == 0:
                        continue
                    
                    pooled_mean = sum(m['avg_belief_correct'] * m['num_samples'] for m in metrics_list) / total_n
                    
                    pooled[agent][opp][role][turn] = {
                        'avg_belief_correct': pooled_mean,
                        'num_samples': total_n
                    }
    return pooled


def print_all_metrics(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    games_per_side = data['config']['games_per_matchup']
    
    # 1. Aggregate Win Rates, Move Times, and Overall Times
    agg = {}
    overall_agent_times = {} 
    
    for r in data['results']:
        it = r['iterations']
        agent = r['agent']
        opp = r['opponent']
        
        if it not in agg: 
            agg[it] = {}
            overall_agent_times[it] = {}
        if agent not in agg[it]: 
            agg[it][agent] = {}
            overall_agent_times[it][agent] = []
        if opp not in agg[it][agent]: 
            agg[it][agent][opp] = {
                'p1_wins': 0, 'p1_draws': 0, 'time_as_p1': 0.0,
                'p2_wins': 0, 'p2_draws': 0, 'time_as_p2': 0.0
            }
            
        if r.get('agent_first', True):
            agg[it][agent][opp]['p1_wins'] = r['wins_p1']
            agg[it][agent][opp]['p1_draws'] = r['draws']
            agg[it][agent][opp]['time_as_p1'] = r['avg_time_p1']
            overall_agent_times[it][agent].append(r['avg_time_p1'])
        else:
            agg[it][agent][opp]['p2_wins'] = r['wins_p2']
            agg[it][agent][opp]['p2_draws'] = r['draws']
            agg[it][agent][opp]['time_as_p2'] = r['avg_time_p2']
            overall_agent_times[it][agent].append(r['avg_time_p2'])

    # 2. Print Iteration-Specific Metrics (Win Rates & Times)
    for it in sorted(agg.keys()):
        print(f"\n\n{'='*80}")
        print(f"==================== ITERATION BUDGET: {it} ====================")
        print(f"{'='*80}")
        
        agents = ['Baseline', 'Det-Only', 'Sel-Only', 'Full OM']
        for agent in agents:
            if agent not in agg[it]: continue
            
            print(f"\n\n{'*'*60}")
            print(f"AGENT: {agent} ({it} Iterations)")
            
            avg_overall_time = sum(overall_agent_times[it][agent]) / len(overall_agent_times[it][agent])
            print(f"OVERALL MASTER MOVE TIME: {avg_overall_time:.5f} sec/move")
            print(f"{'*'*60}")
            
            for opp in ['Random', 'Center-Heavy', 'Corner-Heavy']:
                if opp not in agg[it][agent]: continue
                
                stats = agg[it][agent][opp]
                
                p1_wr = (stats['p1_wins'] / games_per_side) * 100
                p2_wr = (stats['p2_wins'] / games_per_side) * 100
                overall_wr = ((stats['p1_wins'] + stats['p2_wins']) / (games_per_side * 2)) * 100
                
                print(f"\n>>> MATCHUP: {agent} vs {opp} <<<")
                print(f"  [WIN RATES]")
                print(f"    - As Player 1 : {p1_wr:5.2f}%")
                print(f"    - As Player 2 : {p2_wr:5.2f}%")
                print(f"    - OVERALL     : {overall_wr:5.2f}%")
                
                print(f"  [MOVE TIMES]")
                print(f"    - As Player 1 : {stats['time_as_p1']:.5f} sec/move")
                print(f"    - As Player 2 : {stats['time_as_p2']:.5f} sec/move")

    # 3. Print Pooled Belief Convergence at the End
    print(f"\n\n{'='*80}")
    print(f"====== POOLED BELIEF CONVERGENCE: FULL OM (500 & 2000 Iterations) ======")
    print(f"{'='*80}")
    
    belief_analyses = data.get('belief_analyses', {})
    if not belief_analyses:
        print("No belief data found in JSON.")
    else:
        pooled_beliefs = pool_belief_data(belief_analyses)
        agent = 'Full OM'
        
        if agent in pooled_beliefs:
            for opp in ['Random', 'Center-Heavy', 'Corner-Heavy']:
                print(f"\n>>> Full OM Confidence predicting '{opp}' <<<")
                if opp in pooled_beliefs[agent]:
                    for role in ['p1', 'p2']:
                        role_str = "Player 1" if role == 'p1' else "Player 2"
                        if role in pooled_beliefs[agent][opp]:
                            role_data = pooled_beliefs[agent][opp][role]
                            moves = sorted([int(m) for m in role_data.keys()])[:10]
                            
                            b_str_parts = []
                            for m in moves:
                                conf = role_data[str(m)]['avg_belief_correct'] * 100
                                b_str_parts.append(f"T{m}: {conf:05.1f}%")
                                
                            print(f"  [As {role_str}]")
                            print(f"    - {' | '.join(b_str_parts)}")

if __name__ == '__main__':
    filepath = 'data/experiment_results_20260331_163428.json'
    print_all_metrics(filepath)