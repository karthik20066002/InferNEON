#!/usr/bin/env python3
"""E1-lite: demand-only cache replay over drift dumps (preliminary).

Runs every contender on the ref and FLA route traces and reports fused
hit rate + bytes/token per capacity. Caveats (do NOT cite as E1): drift
positions are ~99% prefill tokens, not the decode regime; no prefetch or
admission logic; capacities are expert slots. Answers preliminarily whether
the hybrid beats LS, and shakes out the harness before capture traces land.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.engine import events_from_routed
from sim.eviction import (LRU, LFU, LFUHalving, LeastStale, HelenDual,
                          HybridStaleHeat, Belady, EXPERT_BYTES)

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                   'reports', 'drift_35_top16')
CAPS = (320, 640, 1000, 2000)


def load_trace(kind):
    events = []
    for path in sorted(glob.glob(os.path.join(DIR, f'*_{kind}.json'))):
        d = json.load(open(path))
        for rec in d['records']:
            for li, ids in enumerate(rec['top8']):
                keys = frozenset((li, int(e)) for e in ids)
                # pass = token ordinal across the concatenated stream: one
                # forward pass (40 layer-events) per token. Monotonic, so
                # stale/current tags stay meaningful across prompts.
                events.append((len(events) // 40, keys))
    return events


def policies(cap):
    return [
        LRU(cap),
        LFU(cap),
        LFUHalving(cap),
        LeastStale(cap),
        HelenDual(cap, theta=0.75),
        HelenDual(cap, theta=0.75, halving_events=40),
        HybridStaleHeat(cap, theta=0.50),
        HybridStaleHeat(cap, theta=0.625),
        HybridStaleHeat(cap, theta=0.75),
        Belady(cap),
    ]


def main():
    out = {'label': 'PRELIMINARY (see caveats)', 'capacities': list(CAPS),
           'expert_bytes': EXPERT_BYTES, 'results': {}}
    for kind in ('ref', 'fla'):
        t0 = time.perf_counter()
        events = load_trace(kind)
        out[f'{kind}_positions'] = len(events)
        for cap in CAPS:
            for pol in policies(cap):
                st = pol.run(events)
                n = len(events)
                row = {'hit_rate': st.hit_rate(),
                       'bytes_per_token': pol.bytes_read() / (n / 40),
                       'missed_experts': st.missed_experts}
                out['results'].setdefault(pol.name, {})[f'{kind}@{cap}'] = row
                print(f'{kind} cap={cap} {pol.name:24s} hit={row["hit_rate"]:.4f} '
                      f'B/tok={row["bytes_per_token"]:.0f}', flush=True)
        print(f'{kind}: {len(events)} positions in {time.perf_counter()-t0:.0f}s',
              flush=True)
    with open(os.path.join(DIR, 'e1_lite.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote e1_lite.json')


if __name__ == '__main__':
    main()
