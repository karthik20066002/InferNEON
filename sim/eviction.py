"""Cache/eviction policies as E1 contenders. Pure Python, no torch.

All policies consume fused ReplayEvents (one atomic 8-set demand per
(token, layer)) and score hits by set inclusion. Keys are (layer, expert)
tuples -- experts are per-layer, never a global namespace.

Contenders:
  LRU, LFU                 -- classical baselines (SpecMD expects both weak)
  LFUHalving               -- exponential decay instead of lifetime counts
  LeastStale               -- stale/current pass partition (SpecMD 2026)
  HelenDual                -- exact port of the firmware MFU+LRU design
  HybridStaleHeat          -- LS stale-pool bottom + heat promotion (Helen's)
  Belady                   -- offline optimal ceiling (needs the full trace)
"""
from __future__ import annotations

import math
from collections import OrderedDict

EXPERT_BYTES = 1572864  # 3*2048*512/2 INT4, matches SimConfig


class HeatStore:
    """Decaying call counts + age for heat/retention scores.

    Clock ticks once per demand event. halving_events=None means lifetime
    counts (original firmware semantics); otherwise counts halve every N
    demand events (40 = one token's worth: architecture-derived decay).
    """

    def __init__(self, halving_events=None):
        self.counts: dict = {}
        self.last_used: dict = {}
        self.clock = 0
        self.halving = halving_events
        self.max_count = 0.0

    def tick(self):
        self.clock += 1
        if self.halving and self.clock % self.halving == 0:
            for k in self.counts:
                self.counts[k] *= 0.5
            self.max_count *= 0.5

    def touch(self, key):
        self.counts[key] = self.counts.get(key, 0.0) + 1.0
        self.last_used[key] = self.clock
        if self.counts[key] > self.max_count:
            self.max_count = self.counts[key]

    def age(self, key):
        return self.clock - self.last_used.get(key, self.clock)

    def heat(self, key, log_counts=False):
        c = self.counts.get(key, 0.0)
        if log_counts:
            c = math.log1p(c)
        return c / max(self.age(key), 1)

    def retention(self, key, recency_scale=96.0):
        c = self.counts.get(key, 0.0)
        f = math.log1p(c) / math.log1p(max(self.max_count, 1.0))
        r = 1.0 / (1.0 + self.age(key) / recency_scale)
        return 0.25 * f + 0.75 * r


class PolicyResult:
    def __init__(self):
        self.demands = 0
        self.hits = 0
        self.missed_experts = 0

    def hit_rate(self):
        return self.hits / self.demands if self.demands else 0.0


class Policy:
    name = 'base'

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES):
        if capacity_slots < 8:
            raise ValueError('capacity must hold at least one fused demand (8)')
        self.cap = capacity_slots
        self.expert_bytes = expert_bytes
        self.stats = PolicyResult()

    def demand(self, keys, pass_id):
        """Process one fused 8-set. Returns True on full-set hit."""
        raise NotImplementedError

    def _admit(self, keys):
        raise NotImplementedError

    def run(self, events):
        """events: iterable of (pass_id, frozenset(keys)). Resets stats."""
        self.stats = PolicyResult()
        for pass_id, keys in events:
            hit = self.demand(keys, pass_id)
            self.stats.demands += 1
            if hit:
                self.stats.hits += 1
        return self.stats

    def bytes_read(self):
        return self.stats.missed_experts * self.expert_bytes


class LRU(Policy):
    name = 'LRU'

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES):
        super().__init__(capacity_slots, expert_bytes)
        self.resident: OrderedDict = OrderedDict()

    def demand(self, keys, pass_id):
        need = set(keys)
        hit = need <= set(self.resident)
        if not hit:
            missing = [k for k in sorted(need) if k not in self.resident]
            self.stats.missed_experts += len(missing)
            # The demanded set is pinned during its own demand: eviction
            # must never drop a key this demand needs (it would neither be
            # re-added nor findable afterwards).
            while len(self.resident) + len(missing) > self.cap:
                oldest = next(k for k in self.resident if k not in need)
                del self.resident[oldest]
            for k in missing:
                self.resident[k] = None
        for k in sorted(need):
            self.resident.move_to_end(k)
        return hit

    _admit = demand


class LFU(Policy):
    name = 'LFU'

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES):
        super().__init__(capacity_slots, expert_bytes)
        self.resident = set()
        self.counts: dict = {}
        self.seq: dict = {}
        self._seq = 0

    def _victim(self, protected=frozenset()):
        cands = [k for k in self.resident if k not in protected]
        return min(cands, key=lambda k: (self.counts.get(k, 0), self.seq.get(k, 0)))

    def demand(self, keys, pass_id):
        need = set(keys)
        hit = need <= self.resident
        for k in need:
            self.counts[k] = self.counts.get(k, 0) + 1
            if k not in self.seq:
                self._seq += 1
                self.seq[k] = self._seq
        if not hit:
            missing = sorted(need - self.resident)
            self.stats.missed_experts += len(missing)
            while len(self.resident) + len(missing) > self.cap:
                self.resident.discard(self._victim(protected=need))
            self.resident.update(missing)
        return hit


class LFUHalving(LFU):
    name = 'LFU-halving-40'

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES, period=40):
        super().__init__(capacity_slots, expert_bytes)
        self.name = f'LFU-halving-{period}'
        self.period = period
        self.clock = 0

    def demand(self, keys, pass_id):
        self.clock += 1
        if self.clock % self.period == 0:
            for k in self.counts:
                self.counts[k] *= 0.5
        return super().demand(keys, pass_id)


class LeastStale(Policy):
    """Stale/current pass partition (SpecMD 2026 §4).

    Victims come from stale residents (pass < current), FIFO by
    (pass, seq); current-pass experts are unevictable except on OOM
    (nothing stale left), when the oldest current goes.
    """

    name = 'LeastStale'

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES):
        super().__init__(capacity_slots, expert_bytes)
        self.resident: dict = {}
        self._seq = 0

    def demand(self, keys, pass_id):
        need = set(keys)
        hit = all(k in self.resident for k in need)
        if not hit:
            missing = sorted(need - set(self.resident))
            self.stats.missed_experts += len(missing)
            while len(self.resident) + len(missing) > self.cap:
                stale = [(p, s, k) for k, (p, s) in self.resident.items()
                         if p < pass_id and k not in need]
                if stale:
                    _, _, victim = min(stale)
                else:
                    cands = [k for k in self.resident if k not in need]
                    victim = min(cands,
                                 key=lambda k: (self.resident[k][0], self.resident[k][1]))
                del self.resident[victim]
            for k in missing:
                self._seq += 1
                self.resident[k] = (pass_id, self._seq)
        for k in need:
            if k in self.resident:
                _, s = self.resident[k]
                self.resident[k] = (pass_id, s)
        return hit


class _TieredBase(Policy):
    """Shared halves machinery for HelenDual / HybridStaleHeat.

    Top tier (cap//2): protected hot experts. Bottom tier: everything else.
    HeatStore counters (decaying or lifetime) drive promotion and victims.
    """

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES,
                 theta=0.75, halving_events=None, log_heat=False,
                 recency_scale=96.0):
        super().__init__(capacity_slots, expert_bytes)
        self.cap_top = capacity_slots // 2
        self.cap_bottom = capacity_slots - self.cap_top
        self.theta = theta
        self.store = HeatStore(halving_events)
        self.log_heat = log_heat
        self.recency_scale = recency_scale
        self.top = set()
        self.bottom = set()
        self.passtag: dict = {}

    def _heat(self, key):
        return self.store.heat(key, self.log_heat)

    def _promote(self):
        if len(self.bottom) < self.theta * self.cap_bottom:
            return
        if len(self.top) >= self.cap_top and not self.top:
            return
        cand = max(self.bottom, key=self._heat)
        if not self.top:
            self.bottom.discard(cand)
            self.top.add(cand)
            return
        cold = min(self.top, key=self._heat)
        if self._heat(cand) > self._heat(cold):
            self.bottom.discard(cand)
            self.top.discard(cold)
            self.top.add(cand)
            self.bottom.add(cold)

    def _evict_bottom(self, n, pass_id, need):
        raise NotImplementedError

    def demand(self, keys, pass_id):
        need = set(keys)
        resident = self.top | self.bottom
        hit = need <= resident
        self.store.tick()
        for k in need:
            self.store.touch(k)
            if k in resident:
                self.passtag[k] = pass_id
        if not hit:
            missing = sorted(need - resident)
            self.stats.missed_experts += len(missing)
            while len(self.top) + len(self.bottom) + len(missing) > self.cap:
                self._evict_bottom(1, pass_id, need)
            for k in missing:
                self.bottom.add(k)
                self.passtag[k] = pass_id
        self._promote()
        return hit


class HelenDual(_TieredBase):
    """Exact firmware port: halves split, heat promotion at theta, LRU-tier
    bottom with retention-score eviction (lowest V_e)."""

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES,
                 theta=0.75, halving_events=None, log_heat=False,
                 recency_scale=96.0):
        super().__init__(capacity_slots, expert_bytes, theta, halving_events,
                         log_heat, recency_scale)
        self.name = (f'HelenDual-t{theta}'
                     + ('-halved' if halving_events else '-lifetime')
                     + ('-log' if log_heat else ''))

    def _evict_bottom(self, n, pass_id, need):
        for _ in range(n):
            pool = [k for k in self.bottom if k not in need]
            victim = min(pool,
                         key=lambda k: self.store.retention(k, self.recency_scale))
            self.bottom.discard(victim)
            self.passtag.pop(victim, None)


class HybridStaleHeat(_TieredBase):
    """Helen's hybrid: LS stale-pool bottom + heat promotion.

    Bottom-tier victims come from stale members (pass < current) by lowest
    heat; only with nothing stale does the coldest overall go (OOM rule).
    Top tier identical to HelenDual. Prefetch/demand arrivals enter bottom
    as current-pass members.
    """

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES,
                 theta=0.75, halving_events=40, log_heat=False,
                 recency_scale=96.0):
        super().__init__(capacity_slots, expert_bytes, theta, halving_events,
                         log_heat, recency_scale)
        self.name = (f'HybridStaleHeat-t{theta}'
                     + ('-halved' if halving_events else '-lifetime')
                     + ('-log' if log_heat else ''))

    def _evict_bottom(self, n, pass_id, need):
        for _ in range(n):
            stale = [k for k in self.bottom
                     if self.passtag.get(k, -1) < pass_id and k not in need]
            pool = stale or [k for k in self.bottom if k not in need]
            victim = min(pool, key=self._heat)
            self.bottom.discard(victim)
            self.passtag.pop(victim, None)


class Belady(Policy):
    """Offline optimal ceiling: evict the resident used furthest in future."""

    name = 'Belady'

    def __init__(self, capacity_slots, expert_bytes=EXPERT_BYTES):
        super().__init__(capacity_slots, expert_bytes)
        self._trace = []

    def run(self, events):
        return self.run_with_index(list(events))

    def demand(self, keys, pass_id):
        raise AssertionError('use run_with_index')

    def run_with_index(self, indexed):
        """indexed: list of (pass_id, frozenset). Precomputes next-use."""
        self.stats = PolicyResult()
        positions = [set(ks) for _, ks in indexed]
        next_use = {}
        last = {}
        for i in range(len(positions) - 1, -1, -1):
            for k in positions[i]:
                next_use[(i, k)] = last.get(k, float('inf'))
                last[k] = i
        resident = set()
        for i, (pass_id, keys) in enumerate(indexed):
            need = set(keys)
            hit = need <= resident
            self.stats.demands += 1
            if hit:
                self.stats.hits += 1
                continue
            missing = sorted(need - resident)
            self.stats.missed_experts += len(missing)
            while len(resident) + len(missing) > self.cap:
                cands = [k for k in resident if k not in need]
                if not cands:
                    break
                victim = max(cands, key=lambda k: next_use.get((i, k), float('inf')))
                resident.discard(victim)
            resident.update(missing)
        return self.stats
