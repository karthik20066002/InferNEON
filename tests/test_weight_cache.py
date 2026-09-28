import unittest
import torch
from capture.weight_cache import WeightCache


class WeightCacheTests(unittest.TestCase):
    def test_owned_storage_and_layer_qualified_keys(self):
        cache = WeightCache(64)
        source = torch.arange(4, dtype=torch.float32)
        saved, = cache.get((0, 3), lambda: (source,))
        source.zero_()
        self.assertTrue(torch.equal(saved, torch.arange(4, dtype=torch.float32)))
        def forbidden():
            self.fail('hit must not reload')
        hit, = cache.get((0, 3), forbidden)
        self.assertEqual(saved.data_ptr(), hit.data_ptr())
        other, = cache.get((1, 3), lambda: (torch.ones(4),))
        self.assertFalse(torch.equal(hit, other))
        self.assertEqual(cache.bytes, 32)

    def test_budget_bypass_no_eviction(self):
        cache = WeightCache(16)
        cache.get((0, 0), lambda: (torch.ones(4),))
        cache.get((0, 1), lambda: (torch.ones(4),))
        self.assertEqual(set(cache.entries), {(0, 0)})
        self.assertEqual(cache.bytes, 16)
        self.assertEqual(cache.bypasses, 1)

    def test_disabled_and_invalid_budget(self):
        cache = WeightCache(0)
        cache.get((0, 0), lambda: (torch.ones(4),))
        self.assertEqual(cache.entries, {})
        with self.assertRaises(ValueError):
            WeightCache(-1)


if __name__ == '__main__':
    unittest.main()
