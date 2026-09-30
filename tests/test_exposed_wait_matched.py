"""Input/aggregation/plot guards for the only current experiment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

from scripts.exposed_wait_matched_L import (
    BANDWIDTHS, CAPACITIES, FAMILIES, LOOKAHEADS, POLICIES,
    check_controls, load_prompt, render_svg, run_case, validate_pairs,
)


def capture(tokens=2, kernel='fast'):
    header = {'type': 'header', 'prompt_id': 'sample', 'text': 'fixture',
              'domain': 'chat', 'seed': 0, 'greedy': True,
              'delta_kernels': kernel, 'prompt_tokens': 1, 'decode_target': tokens}
    # Deliberately omit prefill routes: excluded, never warmed or predicted from.
    rows = [header, {'t': 0, 'phase': 'prefill'}]
    rows += [{'t': t + 1, 'phase': 'decode',
              'top8': [list(range(t * 8, t * 8 + 8)) for _ in range(40)]}
             for t in range(tokens)]
    return rows


class MatchedExperimentTests(unittest.TestCase):
    def load(self, rows, kernel='fast'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            return load_prompt(path, kernel)

    def test_decode_only_layer_qualified_fused_demands(self):
        prompt = self.load(capture())
        self.assertEqual(len(prompt['events']), 80)
        self.assertEqual(prompt['input']['decode_tokens'], 2)
        self.assertEqual(prompt['input']['prefill_tokens_excluded'], 1)
        self.assertEqual(prompt['events'][0], (0, frozenset((0, e) for e in range(8))))
        self.assertEqual(prompt['events'][40], (1, frozenset((0, e) for e in range(8, 16))))
        self.assertEqual(len(prompt['input']['sha256']), 64)

    def test_reject_wrong_family_and_incomplete_decode(self):
        with self.assertRaises(ValueError):
            self.load(capture(), 'reference (forced)')
        with self.assertRaises(ValueError):
            self.load(capture()[:-1])

    def test_reject_bad_token_indices_phases_layers_and_sets(self):
        for field, value in [('t', 8), ('phase', 'unknown'),
                             ('top8', [list(range(8))] * 39),
                             ('top8', [[0] * 8] * 40),
                             ('top8', [list(range(7)) + [256]] * 40)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                rows = capture()
                rows[-1][field] = value
                self.load(rows)

    def test_paired_inputs_not_paired_generated_routes(self):
        fast = self.load(capture())
        ref = self.load(capture(kernel='reference (forced)'), 'reference (forced)')
        validate_pairs({'FLA': [fast], 'reference': [ref]})
        ref['header']['text'] = 'different prompt'
        with self.assertRaises(ValueError):
            validate_pairs({'FLA': [fast], 'reference': [ref]})

    def test_pool_wait_by_tokens_and_reset_cache_per_prompt(self):
        one = self.load(capture(tokens=1))
        two = self.load(capture(tokens=2))
        row = run_case([one, two], 'FLA', 100e6, 6500, 'LRU/no hint', None)
        self.assertEqual(row['decode_tokens'], 3)
        self.assertEqual(row['demands'], 120)
        self.assertEqual(row['fetched_bytes'], 3 * 320 * 1572864)
        self.assertAlmostEqual(row['exposed_wait_ms_per_token'], 320 * 1572864 / 100e6 * 1000)
        perfect = run_case([one, two], 'FLA', 100e6, 6500, 'perfect IDs', 0)
        check_controls([row, perfect])
        broken = copy.deepcopy(perfect)
        broken['per_prompt'][0]['exposed_s'] += 1
        with self.assertRaises(AssertionError):
            check_controls([row, broken])

    def test_single_plot_has_matched_panels_and_honest_caption(self):
        variants = [('LRU/no hint', None)] + [(p, lag) for lag in LOOKAHEADS for p in POLICIES[1:]]
        rows = [{'trace_family': f, 'bandwidth_bps': bw, 'capacity_slots': cap,
                 'policy': p, 'lookahead_events': lag, 'exposed_wait_ms_per_token': 1000}
                for f in FAMILIES for bw in BANDWIDTHS for cap in CAPACITIES
                for p, lag in variants]
        svg = render_svg({'results': rows})
        root = ElementTree.fromstring(svg)
        self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}polyline')), 16 * 3)
        self.assertIn('Empty cache', svg)
        self.assertIn('ASSUMED', svg)
        self.assertIn('8 ms/layer event', svg)
        self.assertIn('decode only', svg)


if __name__ == '__main__':
    unittest.main()
