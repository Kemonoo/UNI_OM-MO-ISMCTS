"""
Post-run analysis of the main ablation JSON. Produces:
  * per-matchup win/draw rates (overall, combining both seats) for all agents
  * a two-proportion z-test table for every (agent vs Baseline) and
    (OM-Sel vs OM-Det) matchup at both budgets
    (victory = success, draws/losses = failure)
  * targeted checks for the win-rate, ablation and convergence claims

No third-party stats dependency: the normal CDF is computed via math.erfc.

Usage:  python analyze_results.py <results.json>
"""

import json
import sys
import math
from collections import defaultdict

import naming

AGENTS = naming.STORED_AGENTS  # ["Baseline","Det-Only","Sel-Only","Full-OM"]
OPPONENTS = ["Random", "Center-Heavy", "Corner-Heavy"]
D = naming.disp  # stored key -> display label (OM-X) for printed output


def two_prop_ztest(x1, n1, x2, n2):
    """Pooled two-proportion z-test. Returns (p1, p2, diff_pp, z, p_value_two_sided)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 0.0, 0.0, 0.0, 1.0)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        z = 0.0
    else:
        z = (p1 - p2) / se
    # two-sided p-value from the standard normal survival function
    p_val = math.erfc(abs(z) / math.sqrt(2))
    return (p1 * 100, p2 * 100, (p1 - p2) * 100, z, p_val)


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def build_index(data):
    """agg[(budget, agent, opp)] = dict with combined-seat wins/games/draws + per-seat rates."""
    gpm = data['config']['games_per_matchup']
    agg = defaultdict(lambda: {'wins': 0, 'games': 0, 'draws': 0,
                               'p1_wr': None, 'p2_wr': None})
    for r in data['results']:
        key = (r['iterations'], r['agent'], r['opponent'])
        cell = agg[key]
        if r['agent_first']:
            cell['wins'] += r['wins_p1']
            cell['p1_wr'] = r['p1_winrate']
        else:
            cell['wins'] += r['wins_p2']
            cell['p2_wr'] = r['p2_winrate']
        cell['games'] += gpm
        cell['draws'] += r['draws']
    return agg, gpm


def print_winrate_tables(agg, budgets):
    for b in budgets:
        print(f"\n{'='*78}\nWIN / DRAW RATES  @ {b} iterations  (overall = both seats combined)\n{'='*78}")
        header = f"{'agent':<10}" + "".join(f"{o:>16}" for o in OPPONENTS)
        print(header)
        for agent in AGENTS:
            row = f"{D(agent):<10}"
            for opp in OPPONENTS:
                c = agg.get((b, agent, opp))
                if not c or c['games'] == 0:
                    row += f"{'-':>16}"
                else:
                    wr = c['wins'] / c['games'] * 100
                    dr = c['draws'] / c['games'] * 100
                    row += f"{wr:>9.2f}/{dr:>4.1f}d"
            print(row)
        print("  (format: win% / draw%d ; n = 2 x games_per_matchup per cell)")


def print_ztest_table(agg, budgets, gpm):
    n_side = 2 * gpm
    print(f"\n{'='*78}\nTWO-PROPORTION Z-TESTS  (victory=success; n={n_side}/side)\n{'='*78}")
    comparisons = [("Det-Only", "Baseline"), ("Sel-Only", "Baseline"),
                   ("Full-OM", "Baseline"),
                   ("Sel-Only", "Det-Only")]
    for b in budgets:
        print(f"\n--- {b} iterations ---")
        print(f"{'comparison':<24}{'opponent':<14}{'A%':>8}{'B%':>8}{'diff_pp':>9}{'z':>8}{'p':>10}  sig")
        for (A, B) in comparisons:
            for opp in OPPONENTS:
                cA = agg.get((b, A, opp))
                cB = agg.get((b, B, opp))
                if not cA or not cB:
                    continue
                pA, pB, diff, z, p = two_prop_ztest(cA['wins'], cA['games'], cB['wins'], cB['games'])
                print(f"{D(A)+' vs '+D(B):<24}{opp:<14}{pA:>8.2f}{pB:>8.2f}{diff:>+9.2f}{z:>8.2f}{p:>10.2e}  {stars(p)}")


def claim_checks(agg, budgets, gpm):
    print(f"\n{'='*78}\nTARGETED CLAIM CHECKS\n{'='*78}")

    def wr(b, agent, opp):
        c = agg.get((b, agent, opp))
        return None if not c or c['games'] == 0 else c['wins'] / c['games'] * 100

    # 1. Does OM-Det still UNDERPERFORM baseline at 2000 (notably vs Corner-Heavy)?
    print("\n[1] OM-Det vs Baseline at 2000 iters (underperformance check):")
    if 2000 in budgets:
        for opp in OPPONENTS:
            d, base = wr(2000, "Det-Only", opp), wr(2000, "Baseline", opp)
            if d is None or base is None:
                continue
            cD, cB = agg[(2000, "Det-Only", opp)], agg[(2000, "Baseline", opp)]
            _, _, diff, z, p = two_prop_ztest(cD['wins'], cD['games'], cB['wins'], cB['games'])
            verdict = "UNDERPERFORMS" if diff < 0 and p < 0.05 else ("~equal" if p >= 0.05 else "OUTPERFORMS")
            print(f"    vs {opp:<13}: Det={d:.2f}%  Base={base:.2f}%  diff={diff:+.2f}pp  p={p:.2e}  -> {verdict}")

    # 2. Is Sel-Only the strongest single component? Count matchups it beats Baseline (6 = 3 opp x 2 budget)
    print("\n[2] Sel-Only vs Baseline: matchups won (significant, p<0.05):")
    won = 0
    for b in budgets:
        for opp in OPPONENTS:
            cS, cB = agg.get((b, "Sel-Only", opp)), agg.get((b, "Baseline", opp))
            if not cS or not cB:
                continue
            _, _, diff, z, p = two_prop_ztest(cS['wins'], cS['games'], cB['wins'], cB['games'])
            sig = (p < 0.05)
            if diff > 0 and sig:
                won += 1
            print(f"    {b:>5} vs {opp:<13}: diff={diff:+.2f}pp  p={p:.2e}  {'WIN' if (diff>0 and sig) else ('ns' if not sig else 'LOSE')}")
    print(f"    => Sel-Only significantly beats Baseline in {won}/6 matchups")

    # 3. Peak win rate + draw rates
    print("\n[3] Peak overall win rate per agent (across all opp/budget):")
    for agent in AGENTS:
        best = None
        for b in budgets:
            for opp in OPPONENTS:
                w = wr(b, agent, opp)
                if w is not None and (best is None or w > best[0]):
                    best = (w, b, opp)
        if best:
            print(f"    {D(agent):<10}: peak {best[0]:.2f}%  @ {best[1]} iters vs {best[2]}")

    print("\n[4] Draw rates per matchup (overall):")
    for b in budgets:
        print(f"  --- {b} iters ---")
        for agent in AGENTS:
            parts = []
            for opp in OPPONENTS:
                c = agg.get((b, agent, opp))
                if c and c['games']:
                    parts.append(f"{opp[:4]}={c['draws']/c['games']*100:.1f}%")
            print(f"    {D(agent):<10}: " + "  ".join(parts))


def main(path):
    with open(path) as f:
        data = json.load(f)
    budgets = data['config']['iteration_budgets']
    agg, gpm = build_index(data)
    print(f"Loaded {path}")
    print(f"games_per_matchup={gpm}  budgets={budgets}  n_per_side={2*gpm}")
    print_winrate_tables(agg, budgets)
    print_ztest_table(agg, budgets, gpm)
    claim_checks(agg, budgets, gpm)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python analyze_results.py <results.json>")
        sys.exit(1)
    main(sys.argv[1])
