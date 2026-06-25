"""
belief_argmax.py

RQ3 metric: report the argmax-identification rate (true model assigned the
single highest posterior) alongside the mean posterior on the true model.

Usage:  python belief_argmax.py <results.json>
"""

import json
import sys
from collections import defaultdict

import naming

opp_idx = {"Random": 0, "Center-Heavy": 1, "Corner-Heavy": 2}


def report(path):
    with open(path) as f:
        data = json.load(f)

    # accumulate per (agent, opponent, role, step): mean posterior + argmax-correct rate
    acc = defaultdict(lambda: [0.0, 0, 0])  # sum_posterior, n, argmax_correct
    for r in data['results']:
        agent = r['agent']
        if not r.get('belief_histories'):
            continue
        role = 'P1' if r['agent_first'] else 'P2'
        ti = opp_idx[r['opponent']]
        for e in r['belief_histories']:
            b = e['beliefs']
            step = e['absolute_turn']
            rec = acc[(agent, r['opponent'], role, step)]
            rec[0] += b[ti]
            rec[1] += 1
            mx = max(b)
            if b[ti] == mx and b.count(mx) == 1:
                rec[2] += 1

    for (agent, opp, role, step), (s, n, am) in sorted(acc.items()):
        if step <= 10:
            print(f"{naming.disp(agent):<10} {opp:<13} {role} step{step:>2}: "
                  f"mean_post={s/n:5.3f}  argmax_rate={am/n:5.3f}  (n={n})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python belief_argmax.py <results.json>")
        sys.exit(1)
    report(sys.argv[1])
