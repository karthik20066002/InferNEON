import unittest
from sim.engine import events_from_routed
from sim.eviction import (LRU, LFU, LFUHalving, LeastStale, HelenDual,
                          HybridStaleHeat, Belady, HeatStore)


def evs(lists, start=0):
    return [(start + i, frozenset(s)) for i, s in enumerate(lists)]


class EvictionTests(unittest.TestCase):
    def test_fused_hit_needs_all_eight(self):
        p = LRU(16)
        p.demand(frozenset({1, 2, 3, 4, 5, 6, 7, 8}), 0)
        # 7 of 8 resident is a MISS (fused, not partial credit)
        self.assertFalse(p.demand(frozenset({1, 2, 3, 4, 5, 6, 7, 9}), 1))
        self.assertEqual(p.stats.missed_experts, 8 + 1)

    def test_lru_evicts_oldest(self):
        p = LRU(16)
        p.demand(frozenset(range(8)), 0)
        p.demand(frozenset(range(8, 16)), 1)
        # 0..7 are now LRU-oldest; demand them back plus pressure
        p.demand(frozenset(range(16, 24)), 2)
        self.assertFalse({0} <= set(p.resident))

    def test_capacity_floor(self):
        with self.assertRaises(ValueError):
            LRU(7)

    def test_heat_halving(self):
        s = HeatStore(halving_events=4)
        for _ in range(4):
            s.tick()
            s.touch('a')
        # halve fires at tick 4 before the touch: 3*0.5+1
        self.assertAlmostEqual(s.counts['a'], 2.5)
        for _ in range(4):
            s.tick()
            s.touch('a')
        # ticks 5..7 add 3 (->5.5), tick 8 halves then touches
        self.assertAlmostEqual(s.counts['a'], 3.75)

    def test_leaststale_protects_current_pass(self):
        p = LeastStale(8)
        p.demand(frozenset(range(8)), 0)
        # new pass, new experts, full cache: stale (pass 0) must go first
        self.assertFalse(p.demand(frozenset(range(8, 16)), 1))
        self.assertTrue(all(k not in p.resident for k in range(8)))
        self.assertTrue(all(k in p.resident for k in range(8, 16)))

    def test_hybrid_promotes_hot_to_top(self):
        p = HybridStaleHeat(16, theta=0.5, halving_events=None)
        # fill bottom (cap 8) with cold experts, then heat one of them
        p.demand(frozenset(range(8)), 0)
        # heat expert 0 by demanding sets containing it repeatedly
        for i in range(2, 7):
            p.demand(frozenset([0] + list(range(100 + i * 7, 107 + i * 7))), i)
        self.assertIn(0, p.top)

    def test_belady_never_worse_than_lru_on_toy(self):
        import random
        rng = random.Random(0)
        trace = evs([frozenset(rng.sample(range(24), 8)) for _ in range(30)])
        lru = LRU(24)
        bel = Belady(24)
        self.assertGreaterEqual(bel.run(trace).hit_rate(),
                                lru.run(trace).hit_rate())

    def test_events_from_routed_keys_are_layer_qualified(self):
        evs_out = events_from_routed('p', [[[1, 2, 3, 4, 5, 6, 7, 0]]])
        self.assertEqual(evs_out[0].experts, frozenset({0, 1, 2, 3, 4, 5, 6, 7}))

    def test_layer_namespaces_do_not_collide(self):
        p = LRU(16)
        p.demand(frozenset({(0, 1)}), 0)
        # same expert id, different layer: still a miss
        self.assertFalse(p.demand(frozenset({(1, 1)}), 1))


if __name__ == '__main__':
    unittest.main()
