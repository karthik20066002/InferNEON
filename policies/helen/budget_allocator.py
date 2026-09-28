"""policies.helen.budget_allocator + predictor_scheduler — interfaces + stubs."""
from abc import ABC, abstractmethod

class BudgetAllocator(ABC):
    """Input: per-layer H_l(c) curves from calibration. Output: slots per layer summing to C.
    Deterministic. Must NOT touch traces directly, only the curves + C."""
    @abstractmethod
    def allocate(self, curves: dict, C: int) -> dict: raise NotImplementedError

class EqualSplit(BudgetAllocator):
    def allocate(self, curves: dict, C: int) -> dict:
        L = len(curves); base, rem = divmod(C, L)
        return {l: base + (1 if i < rem else 0) for i, l in enumerate(sorted(curves))}

class PredictorScheduler(ABC):
    """Produces ranked [(layer, expert, confidence)] for a future layer at a
    defined availability time (input-ready + own compute). Must state which
    hidden state it needs (h_in vs h_postmix) and its compute cost."""
    @abstractmethod
    def predict(self, layer: int, hidden_state, router, norm) -> list: raise NotImplementedError

class NullPredictor(PredictorScheduler):
    def predict(self, layer: int, hidden_state, router, norm) -> list: return []
