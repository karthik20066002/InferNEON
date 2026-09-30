"""Single-lane storage queue + LRU resident cache; no hardware timings.

One lane serializes full-expert transfers at an ASSUMED bandwidth.
Each decode event waits for its atomic 8-set, then computes for ASSUMED
slack_s seconds. Stalls delay subsequent events and hint releases: the
clock never returns to an unstalled i*slack_s schedule. Exposed wait is
the sum of those stalls (elapsed time minus N*slack_s), not recall and
not a separate bandwidth allowance for every outstanding hint.

At event i, predict(i + lookahead) is released, with no negative-time
warmup. Demand jumps unsent prefetch, including promotion of a queued
key that becomes demanded. An active transfer is never preempted.
No cancellation, speculative in-flight cap, or worst-case bound exists.

Cache and lane start empty on every call. Keys are (layer, expert).
Fused demands pin their members only while being satisfied, not during
preceding compute. Eviction and simultaneous demand touches use sorted
keys for deterministic LRU ties. Accounting stops when the final demand
is satisfied: trailing hints are not drained or charged as completed.
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque


def run_laned(events, predict, lookahead, capacity_slots, bandwidth_bps,
              slack_s, expert_bytes):
    """Replay fused demands through one storage lane + LRU cache.

    events: [(pass_id, frozenset keys)] in decode order.
    predict(j) -> set of keys claimed for demand j, released at j-lookahead.
    Returns dict(exposed_s, fetched_bytes, wasted_bytes, demands, hits).
    Waste counts completed speculative loads not used before eviction/end,
    per load, even if the same key was demanded earlier in the trace.
    """
    if not isinstance(capacity_slots, int) or capacity_slots < 8:
        raise ValueError('capacity must hold one fused demand (8)')
    if not isinstance(lookahead, int) or lookahead < 0:
        raise ValueError('lookahead must be a nonnegative event count')
    for name, value in [('bandwidth_bps', bandwidth_bps),
                        ('expert_bytes', expert_bytes), ('slack_s', slack_s)]:
        if not math.isfinite(value) or value < 0 or (name != 'slack_s' and value == 0):
            raise ValueError(f'invalid {name}: {value}')
    needs = [frozenset(ks) for _, ks in events]
    if any(len(need) != 8 for need in needs):
        raise ValueError('each fused demand must contain exactly 8 keys')
    n = len(needs)
    ttime = expert_bytes / bandwidth_bps

    resident = OrderedDict()
    queued = set()
    pref_q = deque()  # (key, release); FIFO
    urg_q = deque()   # (key, release); ahead of prefetches
    in_flight = None  # (key, done, was_prefetch) committed past now
    busy_until = 0.0  # lane clock: advances on EVERY transfer
    now = 0.0
    exposed = 0.0

    fetched_bytes = 0
    wasted_bytes = 0
    hits = 0
    unused_prefetch = set()

    def admit(key, protected):
        nonlocal wasted_bytes
        if key in resident:
            resident.move_to_end(key)
            return
        if len(resident) >= capacity_slots:
            victim = next(k for k in resident if k not in protected)
            del resident[victim]
            if victim in unused_prefetch:
                unused_prefetch.remove(victim)
                wasted_bytes += expert_bytes
        resident[key] = None

    def finish(key, done, protected, was_pre):
        nonlocal busy_until, fetched_bytes
        busy_until = done
        queued.discard(key)
        fetched_bytes += expert_bytes
        admit(key, protected)
        if was_pre:
            unused_prefetch.add(key)

    def settle(t, protected, stop=None):
        nonlocal busy_until, in_flight
        while True:
            if in_flight is not None:
                key, done, was_pre = in_flight
                if done > t:
                    busy_until = done
                    return
                finish(key, done, protected, was_pre)
                in_flight = None
                if stop is not None and stop():
                    return
                continue
            src = urg_q if urg_q else (pref_q if pref_q else None)
            if src is None:
                return
            key, rel = src[0]
            was_pre = src is pref_q
            start = busy_until if busy_until > rel else rel
            # At the exact event boundary, demand wins before an unsent hint.
            if start >= t:
                return
            done = start + ttime
            src.popleft()
            if done > t:
                busy_until = done
                in_flight = (key, done, was_pre)
                return
            finish(key, done, protected, was_pre)
            if stop is not None and stop():
                return

    for i, need in enumerate(needs):
        # Finish real overlap since the previous event; do not pin future IDs.
        settle(now, frozenset())
        j = i + lookahead
        if j < n:
            for k in sorted(k for k in predict(j)
                            if k not in resident and k not in queued):
                pref_q.append((k, now))
                queued.add(k)
        missing = sorted(k for k in need if k not in resident)
        if not missing:
            hits += 1
        else:
            # Upgrade queued demands; only an already-started transfer is fixed.
            promote = set(missing) & queued
            if promote:
                pref_q = deque((k, rel) for k, rel in pref_q if k not in promote)
            for k in missing:
                if in_flight is None or k != in_flight[0]:
                    urg_q.append((k, now))
                    queued.add(k)
            settle(float('inf'), need,
                   stop=lambda: all(k in resident for k in missing))
            exposed += busy_until - now
            now = busy_until
        for k in sorted(need):
            resident.move_to_end(k)
            unused_prefetch.discard(k)
        now += slack_s

    wasted_bytes += len(unused_prefetch) * expert_bytes
    return {'exposed_s': exposed, 'fetched_bytes': fetched_bytes,
            'wasted_bytes': wasted_bytes, 'demands': n, 'hits': hits}
