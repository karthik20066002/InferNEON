"""policies.helen.speculation_controller — interface + stub. Rules R1-R5 togglable."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SpecRuleSet:
    R1_hold_spec_in_sw_queue: bool = True
    R2_cap_spec_bytes_in_flight: int = 262144  # one tile ASSUMED
    R3_spec_only_when_no_exact_pending: bool = True
    R4_byte_budget_by_time_to_need: bool = False
    R5_per_layer_throttle_on_precision: bool = False

class SpeculationController(ABC):
    """Decides which (layer, expert, confidence) predictions become fetch
    requests, when released to device, when cancelled in SW queue.
    Submitted device requests can NEVER be cancelled (bytes always counted).
    Deterministic. Must NOT touch eviction internals.
    """
    @abstractmethod
    def filter(self, preds: list, now_ms: float, state: dict, rules: SpecRuleSet) -> list:
        raise NotImplementedError

class NoSpeculation(SpeculationController):
    def filter(self, preds: list, now_ms: float, state: dict, rules: SpecRuleSet) -> list:
        return []
