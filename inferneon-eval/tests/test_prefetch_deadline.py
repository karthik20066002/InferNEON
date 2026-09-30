"""Matched-timing controls replace the retired per-hint overlap arithmetic.

The old test compared unequal L and credited overlapping hints with
independent full bandwidth. It cannot support an exposed-wait claim.
"""
import unittest

from sim.predictors import PerfectIDs, PreviousToken8, no_hint
from sim.queue import run_laned


def events(tokens=3, layers=40):
    return [(t, frozenset((layer, t * 8 + e) for e in range(8)))
            for t in range(tokens) for layer in range(layers)]


class MatchedPredictorTests(unittest.TestCase):
    def test_one_oracle_definition_at_every_lookahead(self):
        trace = events()
        oracle = PerfectIDs(trace)
        for lookahead in (0, 1, 4, 40):
            for i in range(len(trace) - lookahead):
                j = i + lookahead
                self.assertEqual(oracle.predict(j), trace[j][1])

    def test_b1_previous_token_same_layer_not_previous_event(self):
        trace = events()
        b1 = PreviousToken8(trace)
        for j in range(40):
            self.assertEqual(b1.predict(j), frozenset())
        for j in range(40, len(trace)):
            self.assertEqual(b1.predict(j), trace[j - 40][1])
            self.assertNotEqual(b1.predict(j), trace[j - 1][1])

    def test_b1_information_is_available_at_release(self):
        for lookahead in (0, 1, 4, 40):
            for i in range(120 - lookahead):
                j = i + lookahead
                if j >= 40:
                    self.assertLessEqual(j - 40, i)

    def test_b1_resets_between_prompts(self):
        PreviousToken8(events()).predict(80)
        self.assertEqual(PreviousToken8(events()).predict(0), frozenset())

    def test_l0_perfect_ids_equal_demand_fetch(self):
        trace = events(tokens=2)
        kwargs = dict(lookahead=0, capacity_slots=320, bandwidth_bps=100e6,
                      slack_s=8e-3, expert_bytes=1572864)
        base = run_laned(trace, no_hint, **kwargs)
        perfect = run_laned(trace, PerfectIDs(trace).predict, **kwargs)
        self.assertEqual(base, perfect)


if __name__ == '__main__':
    unittest.main()
