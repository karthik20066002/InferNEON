"""Prefetch-deadline inversion test (synthetic existence proof, no torch).

A = oracle predictor: 100% next-demand IDs, lookahead 0 events.
B = stale predictor (B1 analogue: previous demand), mean recall 0.375,
    lookahead 4 events (one synthetic pass early).

Guess (locked in experiments/prefetch_deadline_guess.md BEFORE running):
A wins recall but LOSES on total exposed I/O, because correctly
predicted bytes can only hide behind lookahead time. Ranking inverts:
accuracy ranks A first, deadlines rank B first.

Numbers: B_hz = SimConfig.B, EXPERT_BYTES from sim.eviction,
compute slack 8 ms/event is ASSUMED and test-local. Empty cache,
demand paging, eviction held fixed (not the variable).
"""
import unittest

from sim.engine import SimConfig, events_from_routed
from sim.eviction import EXPERT_BYTES
from sim.metrics import recall_at_k

COMPUTE_PER_EVENT_S = 8e-3  # ASSUMED, test-local (see guess file)
LOOKAHEAD_A = 0
LOOKAHEAD_B = 4


def exposed_s(n_correct, n_total=8, lookahead_events=0):
    """Unhidden seconds for one fused demand.

    Correctly predicted bytes may prefetch during the lookahead window;
    unpredicted bytes are demand-fetched at the event (fully exposed).
    """
    bw = SimConfig.B
    correct_s = n_correct * EXPERT_BYTES / bw
    hidden = min(correct_s, lookahead_events * COMPUTE_PER_EVENT_S)
    unpredicted_s = (n_total - n_correct) * EXPERT_BYTES / bw
    return (correct_s - hidden) + unpredicted_s


class PrefetchDeadlineTests(unittest.TestCase):
    def test_oracle_late_loses_to_stale_early(self):
        demands = [
            frozenset(range(0, 8)),
            frozenset(range(4, 12)),
            frozenset(range(8, 16)),
            frozenset(range(12, 20)),
        ]
        # Fused contract: build through the real constructor (exactly 8).
        E = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8, 9, 10, 11],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [12, 13, 14, 15, 16, 17, 18, 19],
        ]
        top8 = [[e] for e in E]
        evs = events_from_routed('synth', top8)
        self.assertEqual(len(evs), 4)
        self.assertTrue(all(len(e.experts) == 8 for e in evs))

        pred_a = [set(d) for d in demands]  # oracle
        pred_b = [set(range(100, 108))] + [set(d) for d in demands[:-1]]

        rec_a = [recall_at_k(d, sorted(p), 8) for d, p in zip(demands, pred_a)]
        rec_b = [recall_at_k(d, sorted(p), 8) for d, p in zip(demands, pred_b)]
        self.assertEqual(rec_a, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(rec_b, [0.0, 0.5, 0.5, 0.5])
        mean_a = sum(rec_a) / len(rec_a)
        mean_b = sum(rec_b) / len(rec_b)
        self.assertGreater(mean_a, mean_b)  # A wins accuracy

        exp_a = sum(exposed_s(8, lookahead_events=LOOKAHEAD_A) for _ in demands)
        exp_b = sum(
            exposed_s(len(d & p), lookahead_events=LOOKAHEAD_B)
            for d, p in zip(demands, pred_b)
        )
        # Documented magnitudes: A ~125.8 ms, B ~78.6 ms.
        per_expert = EXPERT_BYTES / SimConfig.B
        self.assertAlmostEqual(exp_a, 32 * per_expert, places=9)
        self.assertAlmostEqual(exp_b, 20 * per_expert, places=9)
        self.assertGreater(exp_a, exp_b)  # ...but B wins exposed I/O


if __name__ == '__main__':
    unittest.main()
