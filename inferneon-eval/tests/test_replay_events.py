import unittest
from sim.engine import ReplayEvent, events_from_routed


class ReplayEventTests(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        # router top-8 ids are distinct (validated upstream); a duplicate
        # collapses the set and must raise, never silently proceed as 7.
        with self.assertRaises(ValueError):
            events_from_routed('p', [[[3, 1, 4, 1, 5, 9, 2, 6]]])

    def test_distinct_eight_ok(self):
        evs = events_from_routed('p', [[[8, 7, 6, 5, 4, 3, 2, 1]]])
        self.assertEqual(evs[0].experts, frozenset({1, 2, 3, 4, 5, 6, 7, 8}))
        self.assertEqual((evs[0].prompt_id, evs[0].token_index, evs[0].layer),
                         ('p', 0, 0))

    def test_fragments_rejected_never_split(self):
        with self.assertRaises(ValueError):
            events_from_routed('p', [[[1, 2, 3]]])  # 3, not 8
        with self.assertRaises(ValueError):
            events_from_routed('p', [[[1, 2, 3, 4, 5, 6, 7]]])  # 7, not 8
        with self.assertRaises(ValueError):
            ReplayEvent('p', 0, 0, frozenset({1, 2, 3}))

    def test_frozen(self):
        ev = ReplayEvent('p', 0, 0, frozenset(range(8)))
        with self.assertRaises(AttributeError):
            ev.layer = 3  # type: ignore

    def test_demote_then_evict_when_bottom_dry(self):
        # Bottom holds only demanded members (or nothing) while top is full
        # and a miss needs room: demote coldest-top first (was empty-pool
        # crash). Constructed directly.
        from sim.eviction import HelenDual, HybridStaleHeat
        for cls in (lambda c: HelenDual(c, theta=1.0),
                    lambda c: HybridStaleHeat(c, theta=1.0)):
            p = cls(16)  # top 8, bottom 8
            p.top = set(range(8))
            p.bottom = set()
            for _ in range(3):
                p.store.tick()
                for k in range(8):
                    p.store.touch(k)
            p.demand(frozenset(range(100, 108)), 5)
            self.assertLessEqual(len(p.top) + len(p.bottom), 16)
            self.assertTrue(set(range(100, 108)) <= p.top | p.bottom)


if __name__ == '__main__':
    unittest.main()
