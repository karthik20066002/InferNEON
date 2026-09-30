#!/usr/bin/env python3
"""The locked decode-only matched-L measurement; stdlib only, sequential.

Run from any directory: python scripts/exposed_wait_matched_L.py
Writes reports/exposed_wait_matched_L.{json,svg}. With --warm-prefill,
prime only the cache from that prompt's prefill and write the same-shaped
reports/exposed_wait_matched_L_warm.json (no additional plot).
No fitting, recall scoring, Belady, hardware timing, or new queue primitive.
See PREREGISTRATION.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.engine import events_from_routed
from sim.eviction import EXPERT_BYTES, LRU
from sim.predictors import PerfectIDs, PreviousToken8, no_hint
from sim.queue import run_laned

LOOKAHEADS = (0, 1, 4, 40)
BANDWIDTHS = (100_000_000, 413_000_000)  # ASSUMED bytes/s, not device results
CAPACITIES = (320, 640, 1000, 2000, 3000, 4000, 6500)
SLACK_S = 0.008  # ASSUMED seconds/layer event; prior test-local value
LAYERS = 40
PROMPT_IDS = tuple(f'{domain}_held_{i:02d}'
                   for domain in ('chat', 'code', 'math', 'sum', 'tamil')
                   for i in range(1, 5))
FAMILIES = {'FLA': ('traces', 'fast'),
            'reference': ('traces_ref', 'reference (forced)')}
POLICIES = ('LRU/no hint', 'B1', 'perfect IDs')
SOURCES = ('scripts/exposed_wait_matched_L.py', 'sim/queue.py',
           'sim/predictors.py', 'sim/engine.py', 'sim/eviction.py',
           'PREREGISTRATION.md')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompt(path, expected_kernel, *, warm_prefill=False):
    """Validate a capture; optionally retain unscored prefill for cache priming."""
    with path.open() as stream:
        header = json.loads(next(stream))
        if header.get('type') != 'header' or header.get('prompt_id') != path.stem:
            raise ValueError(f'{path}: wrong header/prompt ID')
        if header.get('delta_kernels') != expected_kernel:
            raise ValueError(f'{path}: wrong trace family')
        decoded = []
        prefilled = []
        prefill = 0
        for expected_t, line in enumerate(stream):
            row = json.loads(line)
            if row['t'] != expected_t:
                raise ValueError(f'{path}: non-contiguous token index')
            if row['phase'] == 'prefill':
                if decoded:
                    raise ValueError(f'{path}: prefill after decode')
                prefill += 1
                if not warm_prefill:
                    continue
            elif row['phase'] != 'decode':
                raise ValueError(f'{path}: unknown phase')
            layers = row.get('top8')
            if not isinstance(layers, list) or len(layers) != LAYERS or any(
                any(type(e) is not int or not 0 <= e < 256 for e in ids)
                for ids in layers
            ):
                raise ValueError(f'{path}: invalid layer count or expert IDs')
            (prefilled if row['phase'] == 'prefill' else decoded).append(layers)
    if not decoded or len(decoded) != header['decode_target']:
        raise ValueError(f'{path}: incomplete decode')
    if prefill != header['prompt_tokens']:
        raise ValueError(f'{path}: incomplete prefill')
    def fused_events(records):
        return [(event.token_index, frozenset((event.layer, e) for e in event.experts))
                for event in events_from_routed(path.stem, records)]

    events = fused_events(decoded)
    return {'header': header, 'events': events,
            'prefill_events': fused_events(prefilled) if warm_prefill else None,
            'input': {'prompt_id': path.stem, 'sha256': sha256(path),
                      'decode_tokens': len(decoded), 'prefill_tokens_excluded': prefill,
                      'demand_events': len(events)}}


def validate_pairs(families):
    """Same held-out inputs, not necessarily identical generated tokens/routes."""
    for fast, ref in zip(families['FLA'], families['reference'], strict=True):
        for field in ('prompt_id', 'text', 'domain', 'seed', 'greedy',
                      'decode_target', 'prompt_tokens'):
            if fast['header'][field] != ref['header'][field]:
                raise ValueError(f'trace families differ in {field}')


def run_case(prompts, family, bandwidth, capacity, policy, lookahead, *, warm_prefill=False):
    per_prompt = []
    for prompt in prompts:
        initial = ()
        if warm_prefill:
            if prompt.get('prefill_events') is None:
                raise ValueError('warm replay requires the prompt prefill')
            # Identical priming for every contender; no hint policy runs here.
            # Cache the immutable LRU order per prompt/capacity, never decode state.
            primed = prompt.setdefault('prefill_caches', {})
            if capacity not in primed:
                lru = LRU(capacity)
                lru.run(prompt['prefill_events'])
                primed[capacity] = tuple(lru.resident)
            initial = primed[capacity]
        events = prompt['events']
        predict = {'LRU/no hint': lambda: no_hint,
                   'B1': lambda: PreviousToken8(events, LAYERS).predict,
                   'perfect IDs': lambda: PerfectIDs(events).predict}[policy]()
        stats = run_laned(events, predict, lookahead or 0, capacity,
                          bandwidth, SLACK_S, EXPERT_BYTES, initial_resident=initial)
        per_prompt.append({'prompt_id': prompt['header']['prompt_id'],
                           'decode_tokens': prompt['input']['decode_tokens'], **stats})
    totals = {key: sum(row[key] for row in per_prompt)
              for key in ('exposed_s', 'fetched_bytes', 'wasted_bytes', 'demands', 'hits')}
    tokens = sum(row['decode_tokens'] for row in per_prompt)
    return {'trace_family': family, 'bandwidth_bps': bandwidth,
            'capacity_slots': capacity, 'policy': policy,
            'lookahead_events': lookahead, 'decode_tokens': tokens, **totals,
            'exposed_wait_ms_per_token': 1000 * totals['exposed_s'] / tokens,
            'per_prompt': per_prompt}


def check_controls(rows):
    """L=0 perfect IDs must equal no hint; all other gains may have either sign."""
    index = {(r['trace_family'], r['bandwidth_bps'], r['capacity_slots'],
              r['policy'], r['lookahead_events']): r for r in rows}
    for row in rows:
        if row['policy'] != 'LRU/no hint':
            continue
        key = (row['trace_family'], row['bandwidth_bps'], row['capacity_slots'])
        perfect = index[(*key, 'perfect IDs', 0)]
        for base_p, perfect_p in zip(row['per_prompt'], perfect['per_prompt'], strict=True):
            for field in ('exposed_s', 'fetched_bytes', 'wasted_bytes', 'demands', 'hits'):
                if not math.isclose(base_p[field], perfect_p[field], rel_tol=1e-12, abs_tol=1e-9):
                    raise AssertionError(f'L=0 control mismatch: {key}, {field}')


def render_svg(report):
    """One faceted exposed-wait plot (no plotting dependency or recall axis)."""
    rows = report['results']
    index = {(r['trace_family'], r['bandwidth_bps'], r['capacity_slots'],
              r['policy'], r['lookahead_events']): r['exposed_wait_ms_per_token']
             for r in rows}
    width, height = 1480, 1180
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
              f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
              '<title id="title">Decode exposed wait at matched lookahead</title>',
              '<desc id="desc">16 panels: columns are L=0,1,4,40 layer events; rows '
              'are FLA and reference at two assumed bandwidths. Each compares LRU/no '
              'hint, B1, and perfect IDs against cache capacity. Lower wait is better. '
              'Empty cache per prompt; bandwidth and slack are assumed.</desc>',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font-family:DejaVu Sans,sans-serif;fill:#18212b} '
              '.tick{font-size:11px}.axis{font-size:12px}.label{font-size:14px;font-weight:bold}</style>']

    def text(x, y, value, cls='axis', extra=''):
        pieces.append(f'<text x="{x}" y="{y}" class="{cls}" {extra}>{escape(str(value))}</text>')

    def line(x1, y1, x2, y2, color, extra=''):
        pieces.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                      f'stroke="{color}" {extra}/>')

    text(70, 32, 'SIMULATED decode exposed wait — matched L, lower is better', 'label')
    text(70, 55, '20 held-out prompts per family · 128 decode tokens/prompt · same LRU cache for all policies')
    colors = {'LRU/no hint': '#242424', 'B1': '#0072b2', 'perfect IDs': '#d55e00'}
    for i, policy in enumerate(POLICIES):
        x = 75 + i * 240
        line(x, 80, x + 32, 80, colors[policy], 'stroke-width="3"')
        text(x + 40, 85, policy)
    text(820, 85, 'X: cache capacity (expert slots) · Y: exposed wait (ms/decode token)')
    ymax = {}
    for bandwidth in BANDWIDTHS:
        maximum = max(r['exposed_wait_ms_per_token'] for r in rows
                      if r['bandwidth_bps'] == bandwidth)
        ymax[bandwidth] = max(100, math.ceil(maximum * 1.04 / 100) * 100)
    row_specs = [(family, bw) for family in FAMILIES for bw in BANDWIDTHS]
    for ri, (family, bandwidth) in enumerate(row_specs):
        top = 148 + ri * 230
        text(70, top - 8, f'{family} · bandwidth {bandwidth / 1e6:g} MB/s (ASSUMED)', 'label')
        for ci, lookahead in enumerate(LOOKAHEADS):
            left, chart_top = 74 + ci * 355, top + 12
            chart_w, chart_h = 292, 152
            maximum = ymax[bandwidth]
            x = lambda cap: left + (cap - CAPACITIES[0]) / (CAPACITIES[-1] - CAPACITIES[0]) * chart_w
            y = lambda wait: chart_top + chart_h * (1 - wait / maximum)
            text(left + chart_w / 2, chart_top - 3, f'L = {lookahead}', 'axis', 'text-anchor="middle"')
            for tick in range(5):
                value = maximum * tick / 4
                yy = y(value)
                line(left, yy, left + chart_w, yy, '#e0e4e8')
                text(left - 7, yy + 4, f'{value:g}', 'tick', 'text-anchor="end"')
            line(left, chart_top, left, chart_top + chart_h, '#808080')
            line(left, chart_top + chart_h, left + chart_w, chart_top + chart_h, '#808080')
            for cap in (320, 1000, 2000, 4000, 6500):
                line(x(cap), chart_top + chart_h, x(cap), chart_top + chart_h + 4, '#808080')
                text(x(cap), chart_top + chart_h + 18, cap, 'tick', 'text-anchor="middle"')
            for policy in POLICIES:
                lag = None if policy == 'LRU/no hint' else lookahead
                points = [(x(cap), y(index[(family, bandwidth, cap, policy, lag)]))
                          for cap in CAPACITIES]
                coordinates = ' '.join(f'{px:.3f},{py:.3f}' for px, py in points)
                # Dashes keep coincident perfect-L0 and no-hint controls visible.
                dash = ' stroke-dasharray="6 4"' if policy == 'perfect IDs' else ''
                pieces.append(f'<polyline points="{coordinates}" fill="none" '
                              f'stroke="{colors[policy]}" stroke-width="2"{dash}/>')
                for px, py in points:
                    pieces.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2" fill="{colors[policy]}"/>')
    captions = [
        'Empty cache, empty lane, and no predictor history at EVERY prompt; decode only, no prefill warmup. No pre-start hints.',
        'ASSUMED: bandwidth 100e6 / 413e6 bytes/s; compute slack 8 ms/layer event; expert payload 1,572,864 bytes; no request overhead.',
        'One non-preemptive storage lane; demand jumps unsent hints. B1 = previous decode token’s 8 at the same layer (first token: no hint).',
        'Perfect IDs: predict(j) = D_j, released at event j−L, exactly the same L as B1. LRU/no hint is repeated as the timing-independent control.',
        'Wait = sum of demand stalls; pooled wait / decode tokens per family. Cold-start cost included. Descriptive simulation, not hardware or recall.'
    ]
    for i, caption in enumerate(captions):
        text(70, 1072 + i * 20, caption)
    pieces.append('</svg>')
    return '\n'.join(pieces) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-prefill', action='store_true',
                        help='prime LRU from own prefill, score decode only, write *_warm.json')
    warm_prefill = parser.parse_args(argv).warm_prefill
    families = {}
    inputs = {}
    for family, (directory, kernel) in FAMILIES.items():
        families[family] = [load_prompt(ROOT / 'data' / directory / f'{pid}.jsonl', kernel,
                                       warm_prefill=warm_prefill) for pid in PROMPT_IDS]
        inputs[family] = [dict(prompt['input'], path=f'data/{directory}/{prompt["header"]["prompt_id"]}.jsonl')
                          for prompt in families[family]]
        if any(p['input']['decode_tokens'] != 128 for p in families[family]):
            raise ValueError('locked experiment requires 128 decode tokens/prompt')
    validate_pairs(families)
    variants = [('LRU/no hint', None)] + [(policy, lag) for lag in LOOKAHEADS for policy in POLICIES[1:]]
    rows = []
    for family, prompts in families.items():
        for bandwidth in BANDWIDTHS:
            for capacity in CAPACITIES:
                for policy, lookahead in variants:
                    row = run_case(prompts, family, bandwidth, capacity, policy, lookahead,
                                   warm_prefill=warm_prefill)
                    rows.append(row)
                    print(f'{family:9s} B={bandwidth:g} cap={capacity:4d} '
                          f'{policy:12s} L={str(lookahead):4s} '
                          f'wait={row["exposed_wait_ms_per_token"]:.6f} ms/token', flush=True)
    check_controls(rows)
    report = {
        'schema_version': 1,
        'label': 'SIMULATED, descriptive; bandwidth and slack ASSUMED',
        'metric': 'pooled exposed wait: sum of sequential decode demand stalls / decode tokens',
        'oracle_definition': 'predict(j) = D_j (demand-time perfect IDs); not Belady eviction',
        'config': {
            'trace_families': FAMILIES, 'prompt_ids': PROMPT_IDS, 'split': 'held-out only',
            'phase': 'decode only', 'layers_per_token': LAYERS, 'experts_per_event': 8,
            'lookahead_events': LOOKAHEADS, 'capacity_slots': CAPACITIES,
            'bandwidth_bps': {'values': BANDWIDTHS, 'status': 'ASSUMED'},
            'slack_s_per_event': {'value': SLACK_S, 'status': 'ASSUMED', 'source': 'prior test-local 8ms, not fitted'},
            'expert_bytes': {'value': EXPERT_BYTES, 'status': 'ASSUMED INT4 payload; excludes metadata'},
            'request_overhead_s': {'value': 0, 'status': 'ASSUMED'},
            'cache': ('ASSUMED prefill-warm LRU per prompt at the scored capacity: '
                      'replay own recorded prefill in token-then-layer order, demand-only; '
                      'carry resident keys and LRU order, not time/bytes/counters, into decode; '
                      'idle lane and unchanged empty B1 history at decode start; no cross-prompt state; '
                      'fused demand pinned only while being satisfied'
                      if warm_prefill else
                      'empty LRU per prompt; fused demand pinned only while being satisfied'),
            'lane': 'one non-preemptive transfer; demand promotes unsent hints; no cancellation or speculative cap',
            'hint_release': 'target j released at start of event j-L, after routing; no release before event 0',
            'b1': 'D_(j-40), same layer, or empty for first decode token; no prefill/history across prompts',
            'final_accounting': 'through last demand satisfaction; trailing hints not drained',
            'aggregation': 'sum exposed_s / sum decode_tokens; trace families never pooled together',
        },
        'provenance': {
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {path: sha256(ROOT / path) for path in SOURCES},
            'inputs': inputs,
        },
        'checks': {'perfect_L0_equals_no_hint_per_prompt': True},
        'results': rows,
    }
    suffix = '_warm' if warm_prefill else ''
    target = ROOT / 'reports' / f'exposed_wait_matched_L{suffix}.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)
    if not warm_prefill:
        target.with_suffix('.svg').write_text(render_svg(report))
    print(f'Wrote {target.relative_to(ROOT)}' +
          ('' if warm_prefill else f' and {target.with_suffix(".svg").relative_to(ROOT)}'), flush=True)


if __name__ == '__main__':
    main()
