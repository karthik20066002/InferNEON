#!/usr/bin/env python3
"""Aggregate per-prompt kernel-sensitivity dumps into the citable noise floor.

Pairs <dir>/<pid>_{ref,fla}.json, pools (prompt, token, layer) positions
across all prompts, and reports the SET-drift F locked in
PREREGISTRATION.md, plus per-domain breakdowns and slot-drift diagnostics.
Also reports the p8-p9 probability-gap distributions per domain when dumps
carry top16 (mechanism probe for flip-prone routing, amendment 2026-09-29).

Usage: summarize_drift.py [dir]  (default reports/drift_35)
"""
import glob
import json
import os
import sys
from collections import defaultdict

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'drift_35')


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
    gaps = defaultdict(list)  # domain -> list of (p8-p9) full-softmax gaps
    for pid, ref_path, fla_path in pairs:
        a = json.load(open(ref_path))
        b = json.load(open(fla_path))
        assert a.get('prompt_id') == b.get('prompt_id') == pid, pid
        assert a.get('deterministic_algorithms') and b.get('deterministic_algorithms'), pid
        assert a.get('delta_kernels') != b.get('delta_kernels'), (pid, 'same kernel path')
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
            if 'top16_probs' in ra:
                dom = a.get('domain') or 'unknown'
                for probs in ra['top16_probs']:
                    gaps[dom].append(probs[7] - probs[8])
        per_prompt[pid] = {'positions': p_pos, 'set_drift': p_set / p_pos}
        dom = a.get('domain') or 'unknown'
        per_domain[dom][0] += p_pos
        per_domain[dom][1] += p_set
    import statistics
    gap_stats = {d: {'n': len(v), 'median': statistics.median(v),
                     'mean': statistics.mean(v),
                     'frac_below_1e3': sum(1 for g in v if g < 1e-3) / len(v)}
                 for d, v in sorted(gaps.items())} if gaps else {}
    summary = {
        'label': 'MEASURED',
        'pairs': len(pairs),
        'positions': pos,
        'set_drift_positions': setd,
        'F_set_drift': setd / pos,
        'slot_drift': slots / tslots,
        'per_domain_set_drift': {d: v[1] / v[0] for d, v in sorted(per_domain.items())},
        'p8_p9_gap': gap_stats,
        'per_prompt': per_prompt,
    }
    with open(os.path.join(DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'pairs: {len(pairs)}, positions: {pos}')
    print(f'F (set-drift, THE floor): {setd}/{pos} = {100 * setd / pos:.2f}%')
    print(f'slot-drift (diagnostic): {100 * slots / tslots:.2f}%')
    for d, v in sorted(per_domain.items()):
        print(f'  {d}: {100 * v[1] / v[0]:.2f}%')
    for d, g in sorted(gap_stats.items()):
        print(f'  gap p8-p9 {d}: n={g["n"]} median={g["median"]:.2e} '
              f'mean={g["mean"]:.2e} frac<1e-3={g["frac_below_1e3"]:.3f}')
    print(f'wrote {os.path.join(DIR, "summary.json")}')


if __name__ == '__main__':
    main()
