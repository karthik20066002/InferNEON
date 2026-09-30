"""policies.helen.lfu_halving — interface + stub. Real decayed-LFU is Helen's."""
from abc import ABC, abstractmethod

class EvictionPolicy(ABC):
    """Per-(layer,expert) residency of 10240 keys, capacity in slots.
    Must NOT touch: storage, predictor, admission directly. Deterministic.
    Shared expert (always-active) is NEVER counted as hit/miss/slot.
    """
    @abstractmethod
    def hit(self, layer: int, expert: int) -> bool: raise NotImplementedError
    @abstractmethod
    def insert(self, layer: int, expert: int) -> int | None: raise NotImplementedError
    @abstractmethod
    def touch(self, layer: int, expert: int) -> None: raise NotImplementedError

class NoOpEviction(EvictionPolicy):
    """Stub: infinite cache (for byte-conservation tests only, never for results)."""
    def __init__(self): self.s = set()
    def hit(self, layer, expert): return (layer, expert) in self.s
    def insert(self, layer, expert): self.s.add((layer, expert)); return None
    def touch(self, layer, expert): pass
