"""Capture-only immutable expert tensors; unrelated to simulated policies.

First-touch retention up to an explicit byte budget, no eviction or tuning.
Once full, misses simply stream as before. Callers MUST NOT modify entries.
"""
from collections.abc import Callable

import torch


class WeightCache:
    def __init__(self, max_bytes: int = 0):
        if max_bytes < 0:
            raise ValueError('cache budget must be nonnegative')
        self.max_bytes = max_bytes
        self.entries: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        self.bytes = 0
        self.hits = self.misses = self.bypasses = 0

    def get(self, key: tuple[int, int], loader: Callable):
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        self.misses += 1
        tensors = tuple(loader())
        size = sum(t.numel() * t.element_size() for t in tensors)
        if size <= self.max_bytes - self.bytes:
            # Own independent storage: never retain a view into reusable fused
            # expert buffers or a source mmap. These are read-only to callers.
            tensors = tuple(t.detach().clone().contiguous() for t in tensors)
            self.entries[key] = tensors
            self.bytes += size
            return self.entries[key]
        self.bypasses += 1
        return tensors

    def stats(self):
        return dict(max_bytes=self.max_bytes, bytes=self.bytes,
                    entries=len(self.entries), hits=self.hits,
                    misses=self.misses, bypasses=self.bypasses)
