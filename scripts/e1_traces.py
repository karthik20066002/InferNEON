#!/usr/bin/env python3
"""E1 replay on decode-capture traces (single trace family; preliminary).

Replays data/traces/*.jsonl fused demands through every contender and
reports hit rate + bytes/token per capacity. Labels PRELIMINARY until the
reference-path capture lands and the dual-trace win rule can be applied:
single-trace numbers cannot declare winners per PREREGISTRATION.md.
Capacities span the brief's 320-6500 slots range.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.eviction import (LRU, LFU, LFUHalving, LeastStale, HelenDual,
                          HybridStaleHeat, Belady, EXPERT_BYTES)

TRACES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                      'data', 'traces')
CAPS = (320, 640, 1000, 2000, 3000, 4000, 6500)


def load_events():
    events = []
    files = sorted(glob.glob(os.path.join(TRACES, '*.jsonl')))
    assert files, f'no traces in {TRACES}'
    for path in files:
        recs = []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get('type') == 'header':
                continue
            recs.append(d)
        for rec in recs:
            for li, ids in enumerate(rec['top8']):
                keys = frozenset((li, int(e)) for e in ids)
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
    t0 = time.perf_counter()
    events = load_events()
    toks = len(events) / 40
    print(f'{len(events)} positions, {toks:.0f} tokens', flush=True)
    out = {'label': 'PRELIMINARY single-trace (no dual-trace verdict)',
           'capacities': list(CAPS), 'expert_bytes': EXPERT_BYTES,
           'positions': len(events), 'results': {}}
    for cap in CAPS:
        for pol in policies(cap):
            st = pol.run(events)
            row = {'hit_rate': st.hit_rate(),
                   'bytes_per_token': pol.bytes_read() / toks,
                   'missed_experts': st.missed_experts}
            out['results'].setdefault(pol.name, {})[str(cap)] = row
            print(f'cap={cap:5d} {pol.name:24s} hit={row["hit_rate"]:.4f} '
                  f'B/tok={row["bytes_per_token"]:.0f}', flush=True)
    with open('reports/e1_traces.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'done in {time.perf_counter()-t0:.0f}s; wrote reports/e1_traces.json',
          flush=True)


if __name__ == '__main__':
    main()
