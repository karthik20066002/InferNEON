#!/usr/bin/env python3
"""Aggregate per-prompt kernel-sensitivity dumps into the citable noise floor.

Pairs reports/drift_35/<pid>_{ref,fla}.json, pools (prompt, token, layer)
positions across all prompts, and reports the SET-drift F locked in
PREREGISTRATION.md, plus per-domain breakdowns and slot-drift diagnostics.
"""
import glob
import json
import os
from collections import Counter, defaultdict

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'drift_35')


def main():
    pairs = []
    for fla_path in sorted(glob.glob(os.path.join(DIR, '*_fla.json'))):
        pid = os.path.basename(fla_path)[:-len('_fla.json')]
        ref_path = os.path.join(DIR, pid + '_ref.json')
        if os.path.exists(ref_path):
            pairs.append((pid, ref_path, fla_path))
    if not pairs:
        raise SystemExit(f'no complete ref/fla pairs in {DIR}')
    pos = setd = slots = tslots = 0
    per_domain = defaultdict(lambda: [0, 0])  # domain -> [positions, set_drift]
    per_prompt = {}
    for pid, ref_path, fla_path in pairs:
        a = json.load(open(ref_path))
        b = json.load(open(fla_path))
        assert a.get('prompt_id') == b.get('prompt_id') == pid, pid
        assert a.get('deterministic_algorithms') and b.get('deterministic_algorithms'), pid
        p_pos = p_set = 0
        for ra, rb in zip(a['records'], b['records']):
            for ta, tb in zip(ra['top8'], rb['top8']):
                pos += 1
                p_pos += 1
                if set(ta) != set(tb):
                    setd += 1
                    p_set += 1
                for x, y in zip(ta, tb):
                    tslots += 1
                    if x != y:
                        slots += 1
        per_prompt[pid] = {'positions': p_pos, 'set_drift': p_set / p_pos}
        dom = a.get('domain') or 'unknown'
        per_domain[dom][0] += p_pos
        per_domain[dom][1] += p_set
    summary = {
        'label': 'MEASURED',
        'pairs': len(pairs),
        'positions': pos,
        'set_drift_positions': setd,
        'F_set_drift': setd / pos,
        'slot_drift': slots / tslots,
        'per_domain_set_drift': {d: v[1] / v[0] for d, v in sorted(per_domain.items())},
        'per_prompt': per_prompt,
    }
    with open(os.path.join(DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'pairs: {len(pairs)}, positions: {pos}')
    print(f'F (set-drift, THE floor): {setd}/{pos} = {100 * setd / pos:.2f}%')
    print(f'slot-drift (diagnostic): {100 * slots / tslots:.2f}%')
    for d, v in sorted(per_domain.items()):
        print(f'  {d}: {100 * v[1] / v[0]:.2f}%')
    print('wrote reports/drift_35/summary.json')


if __name__ == '__main__':
    main()
