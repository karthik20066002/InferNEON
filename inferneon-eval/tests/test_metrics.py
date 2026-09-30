import unittest
from sim.metrics import (recall_at_k, exact_set_accuracy, mean_recall,
                         bootstrap_delta_ci, bootstrap_delta_p, holm_reject,
                         wobble, beats_baseline)


class MetricsTests(unittest.TestCase):
    def test_recall_at_k_set_based_rank_blind(self):
        routed = {1, 2, 3, 4, 5, 6, 7, 8}
        self.assertEqual(recall_at_k(routed, [8, 7, 6, 5, 4, 3, 2, 1], 8), 1.0)
        self.assertEqual(recall_at_k(routed, [1, 2, 3, 4, 5, 6, 7, 9], 8), 7 / 8)
        self.assertEqual(recall_at_k(routed, [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3], 12), 1.0)
        self.assertEqual(recall_at_k(routed, [9, 9, 9, 1, 2, 3, 4, 5], 8), 5 / 8)
        with self.assertRaises(ValueError):
            recall_at_k(set(), [1], 8)

    def test_exact_set_accuracy(self):
        routed = {1, 2, 3, 4, 5, 6, 7, 8}
        self.assertEqual(exact_set_accuracy(routed, [8, 7, 6, 5, 4, 3, 2, 1]), 1.0)
        self.assertEqual(exact_set_accuracy(routed, [1, 2, 3, 4, 5, 6, 7, 9]), 0.0)

    def test_mean_recall_pooled(self):
        self.assertAlmostEqual(mean_recall({'p1': [1.0, 0.5], 'p2': [0.0]}), 0.5)

    def test_bootstrap_ci_deterministic_and_sane(self):
        a = {f'p{i}': [0.9, 0.8] for i in range(6)}
        b = {f'p{i}': [0.5, 0.4] for i in range(6)}
        lo, hi = bootstrap_delta_ci(a, b)
        lo2, hi2 = bootstrap_delta_ci(a, b)
        self.assertEqual((lo, hi), (lo2, hi2))
        self.assertGreater(lo, 0)  # clear win: CI excludes zero
        self.assertLess(lo, 0.45)
        self.assertGreater(hi, 0.35)
        # identical policies: CI covers zero
        lo0, hi0 = bootstrap_delta_ci(a, a)
        self.assertLessEqual(lo0, 0)
        self.assertGreaterEqual(hi0, 0)
        with self.assertRaises(ValueError):
            bootstrap_delta_ci({'p1': [1.0]}, {'p1': [1.0, 0.5]})

    def test_beats_baseline_rule(self):
        ok = beats_baseline(0.05, 0.04, (0.01, 0.09), (0.005, 0.08))
        self.assertTrue(ok['verdict'])
        # fails wobble: min margin smaller than cross-trace gap
        noisy = beats_baseline(0.05, 0.01, (0.01, 0.09), (0.001, 0.05))
        self.assertFalse(noisy['exceeds_wobble'])
        self.assertFalse(noisy['verdict'])
        # fails sign on one trace
        neg = beats_baseline(0.05, -0.01, (0.01, 0.09), (-0.05, 0.03))
        self.assertFalse(neg['verdict'])
        # fails CI
        ci = beats_baseline(0.05, 0.04, (-0.01, 0.11), (0.005, 0.08))
        self.assertFalse(ci['ci_excludes_zero'])
        self.assertFalse(ci['verdict'])

    def test_wobble(self):
        self.assertAlmostEqual(wobble(0.9, 0.87), 0.03)

    def test_bootstrap_p_and_holm(self):
        a = {f'p{i}': [0.9, 0.8] for i in range(6)}
        b = {f'p{i}': [0.5, 0.4] for i in range(6)}
        p = bootstrap_delta_p(a, b)
        self.assertEqual(p, bootstrap_delta_p(a, b))  # deterministic
        self.assertLess(p, 0.01)  # clear separation: tiny p
        p0 = bootstrap_delta_p(a, a)
        self.assertEqual(p0, 1.0)  # identical: every replicate delta is 0
        # Holm: rejects 0.001, stops at 0.03 (0.03 > 0.05/2)
        self.assertEqual(holm_reject([0.001, 0.03, 0.5]), [True, False, False])
        self.assertEqual(holm_reject([0.5, 0.4]), [False, False])
        self.assertEqual(holm_reject([]), [])
        with self.assertRaises(ValueError):
            bootstrap_delta_p({'p1': [1.0]}, {'p2': [1.0]})


if __name__ == '__main__':
    unittest.main()
