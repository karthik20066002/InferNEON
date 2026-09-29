#!/usr/bin/env python3
"""Separate tail packing (A) from bigger noise (B) for domain drift gaps.

Raw router logits are identified only up to a per-position additive
constant (we store full-softmax probs), so per-expert |logit_ref - logit_fla|
is meaningless raw. Instead, per position and per expert in the top-16
INTERSECTION of both traces: d = log(p_ref) - log(p_fla), centered by the
position median (kills the normalizer shift). mean|centered d| per domain =
perturbation magnitude in logit units, normalizer-free.

- Same magnitude across domains -> A (tail packing), paragraph stands.
- Tamil larger -> B is at least part of it, paragraph must say so.

Also rebins the flip-rate-vs-gap curves in logit units (log p8 - log p9),
since probability gaps depend on the softmax normalizer.

Usage: noise_magnitude.py [dir]  (default reports/drift_35_top16)
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'drift_35_top16')

EDGES = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.0]
LOG_EDGES = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0]


def grid(vals):
    return sorted(vals)


def main():
    pert = defaultdict(list)  # domain -> per-position magnitudes
    bins = defaultdict(lambda: [0, 0])  # (domain, logbin) -> [n, flips]
    pairs = 0
    for ref_path in sorted(glob.glob(os.path.join(DIR, '*_ref.json'))):
        pid = os.path.basename(ref_path)[:-len('_ref.json')]
        fla_path = os.path.join(DIR, pid + '_fla.json')
        if not os.path.exists(fla_path):
            continue
        pairs += 1
        a = json.load(open(ref_path))
        b = json.load(open(fla_path))
        dom = a.get('domain') or 'unknown'
        for ra, rb in zip(a['records'], b['records']):
            if 'top16_probs' not in ra:
                raise SystemExit(f'{pid}: dumps lack top16; rerun with current dump_routes.py')
            for ta, tb, pa, pb, ia, ib in zip(ra['top8'], rb['top8'],
                                              ra['top16_probs'], rb['top16_probs'],
                                              ra['top16_ids'], rb['top16_ids']):
                ma = dict(zip(ia, pa))
                mb = dict(zip(ib, pb))
                inter = [e for e in set(ia) & set(ib)
                         if ma[e] > 0 and mb[e] > 0]
                ds = sorted(math.log(ma[e]) - math.log(mb[e]) for e in inter)
                med = ds[len(ds) // 2]
                pert[dom].append(sum(abs(d - med) for d in ds) / len(ds))
                # logit-units gap rebinning (normalizer cancels in the difference)
                lgap = math.log(pa[7]) - math.log(pa[8])
                bi = next(i for i in range(len(LOG_EDGES) - 1)
                          if LOG_EDGES[i] <= lgap < LOG_EDGES[i + 1])
                bins[(dom, bi)][0] += 1
                if set(ta) != set(tb):
                    bins[(dom, bi)][1] += 1
    print(f'pairs: {pairs}')
    print('perturbation magnitude mean|centered dlogp| per domain (logit units):')
    for dom in grid(pert):
        v = pert[dom]
        print(f'  {dom}: n={len(v)} mean={statistics.mean(v):.3e} '
              f'median={statistics.median(v):.3e}')
    print('flip-rate rebinned by LOGIT gap (edges: <0.001 <0.003 <0.01 <0.03 <0.1 <0.3 <1.0):')
    for dom in grid({d for d, _ in bins}):
        row = []
        for i in range(len(LOG_EDGES) - 1):
            n, f = bins[(dom, i)]
            row.append('%.0f%%' % (100 * f / n) if n else '-')
        print(f'  {dom} ' + ' '.join(row))


if __name__ == '__main__':
    main()
