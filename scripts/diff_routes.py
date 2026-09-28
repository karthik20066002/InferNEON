#!/usr/bin/env python3
"""Diff two route dumps (e.g. reference vs fast DeltaNet kernels).

Reports how many (token, layer) positions change expert sets when only the
kernel path differs: a cheap early signal for quantization tolerance. If a
kernel swap flips many routes, int4-level perturbations will too.
"""
import json
import sys
from collections import Counter


def load(path):
    return json.loads(open(path).read())


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    print(f'A: {a_path} kernels={a.get("delta_kernels")} logits={a["logits_sha256"][:12]} '
          f'det={a.get("deterministic_algorithms")} prompt={a.get("prompt_id")}')
    print(f'B: {b_path} kernels={b.get("delta_kernels")} logits={b["logits_sha256"][:12]} '
          f'det={b.get("deterministic_algorithms")} prompt={b.get("prompt_id")}')
    if a.get('prompt_id') != b.get('prompt_id'):
        raise SystemExit('dumps cover different prompts; not comparable')
    if a['prompt_tokens'] != b['prompt_tokens'] or len(a['records']) != len(b['records']):
        raise SystemExit('dumps cover different token counts; not comparable')
    if a['tokens'] != b['tokens']:
        print(f'NOTE: decoded tokens differ {a["tokens"]} vs {b["tokens"]} '
              f'(greedy paths diverged); comparing shared prefix positions only.')
    n = total = 0
    per_layer = Counter()
    hist = Counter()
    eighth_only = order_only = 0
    for ri, (ra, rb) in enumerate(zip(a['records'], b['records'])):
        for li, (ta, tb) in enumerate(zip(ra['top8'], rb['top8'])):
            total += 1
            if ta != tb:
                n += 1
                per_layer[li] += 1
                sa, sb = set(ta), set(tb)
                nd = 8 - len(sa & sb)
                hist[nd] += 1
                if nd == 0:
                    order_only += 1
                if ta[:7] == tb[:7] and ta[7] != tb[7]:
                    eighth_only += 1
    print(f'route positions: {total}, mismatched: {n} ({100.0 * n / total:.1f}%)')
    print(f'expert-diff histogram (differing experts -> count): {dict(sorted(hist.items()))}')
    print(f'eighth-only: {eighth_only}, order-only: {order_only}')
    print(f'per-layer mismatch counts: {dict(sorted(per_layer.items()))}')


if __name__ == '__main__':
    main()
