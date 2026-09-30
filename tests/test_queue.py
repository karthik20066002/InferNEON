"""Single-lane replay invariants; synthetic byte/second units, no hardware claims."""
import unittest

from sim.eviction import LRU
from sim.queue import run_laned


def demand(start, layer=0):
    return frozenset((layer, e) for e in range(start, start + 8))


def replay(needs, predict=None, lookahead=0, capacity=16, slack=0):
    return run_laned(list(enumerate(needs)), predict or (lambda j: set()),
                     lookahead, capacity, 1, slack, 1)


class QueueTests(unittest.TestCase):
    def test_no_hint_stalls_do_not_run_clock_backwards(self):
        stats = replay([demand(0), demand(8), demand(16)], slack=4)
        self.assertEqual(stats['fetched_bytes'], 24)
        self.assertAlmostEqual(stats['exposed_s'], 24)

    def test_perfect_at_zero_is_no_hint(self):
        needs = [demand(0), demand(4), demand(8), demand(0)]
        for slack in (0, 0.25, 4, 20):
            base = replay(needs, capacity=8, slack=slack)
            perfect = replay(needs, lambda j: needs[j], capacity=8, slack=slack)
            self.assertEqual(perfect['fetched_bytes'], base['fetched_bytes'])
            self.assertAlmostEqual(perfect['exposed_s'], base['exposed_s'])

    def test_no_hint_matches_lru_fused_bytes_and_hits(self):
        needs = [demand(i) for i in (0, 5, 8, 0, 10, 1, 4, 0)]
        for cap in (8, 12, 16, 24):
            lru = LRU(cap, expert_bytes=1)
            lru.run(list(enumerate(needs)))
            stats = replay(needs, capacity=cap, slack=3)
            self.assertEqual(stats['fetched_bytes'], lru.bytes_read())
            self.assertEqual(stats['hits'], lru.stats.hits)
            self.assertAlmostEqual(stats['exposed_s'], lru.bytes_read())

    def test_one_lane_prefetch_can_use_compute_slack_only_once(self):
        needs = [demand(0), demand(8), demand(16)]
        stats = replay(needs, lambda j: needs[j], lookahead=1, capacity=24, slack=4)
        # Cold first demand waits 8; each next demand hides 4 of its 8 reads.
        self.assertAlmostEqual(stats['exposed_s'], 16)
        self.assertEqual(stats['fetched_bytes'], 24)

    def test_demand_promotes_unsent_hint_but_cannot_preempt_active_read(self):
        a, b = demand(0), demand(100)
        # At t=8 A is satisfied. A wrong prefetch starts during 0.5s compute.
        # At t=8.5 B jumps all UNSENT wrong reads, but waits 0.5s for the active one.
        stats = replay([a, b], lambda j: demand(20) | b if j == 1 else set(),
                       lookahead=1, capacity=32, slack=0.5)
        self.assertAlmostEqual(stats['exposed_s'], 16.5)
        self.assertEqual(stats['fetched_bytes'], 17)
        self.assertEqual(stats['wasted_bytes'], 1)

    def test_no_prefetch_starts_at_exact_demand_boundary(self):
        stats = replay([demand(0), demand(100)], lambda j: demand(20),
                       lookahead=1, slack=1)
        self.assertAlmostEqual(stats['exposed_s'], 16)
        self.assertEqual(stats['fetched_bytes'], 17)

    def test_evicted_prefetch_that_was_used_in_past_is_still_wasted(self):
        # A is demanded, then evicted by B. A speculative reload is never used.
        a, b = demand(0), demand(8)
        stats = replay([a, b, b, b], lambda j: a if j == 3 else set(),
                       lookahead=1, capacity=8, slack=1)
        self.assertEqual(stats['wasted_bytes'], 1)
        self.assertEqual(stats['fetched_bytes'], 18)

    def test_empty_trace(self):
        stats = replay([])
        self.assertEqual(stats['demands'], 0)
        self.assertEqual(stats['exposed_s'], 0)

    def test_reject_invalid_inputs(self):
        args = dict(events=[(0, demand(0))], predict=lambda j: set(),
                    lookahead=0, capacity_slots=8, bandwidth_bps=1,
                    slack_s=0, expert_bytes=1)
        for key, value in [('lookahead', -1), ('lookahead', 0.5),
                           ('capacity_slots', 7), ('bandwidth_bps', 0),
                           ('bandwidth_bps', float('inf')), ('slack_s', -1),
                           ('expert_bytes', 0), ('events', [(0, demand(0) - {(0, 0)})])]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                run_laned(**(args | {key: value}))


if __name__ == '__main__':
    unittest.main()
