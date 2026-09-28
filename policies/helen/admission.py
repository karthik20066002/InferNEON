"""policies.helen.admission — interface + trivial stub. Helen writes the real policy."""
from abc import ABC, abstractmethod

class AdmissionPolicy(ABC):
    """Called after a missed expert is fetched to staging and used.

    Inputs: expert key (layer, expert), residency state summary, stats snapshot.
    Outputs: True=retain (evict victim per eviction policy), False=discard.
    When: exactly once per miss-use. Must NOT touch: storage queues, predictor,
    other layers' slots directly. Deterministic given seed + inputs.
    """
    @abstractmethod
    def decide(self, layer: int, expert: int, ctx: dict) -> bool:
        raise NotImplementedError

class AlwaysAdmit(AdmissionPolicy):
    """Trivial reference stub so harness runs end to end."""
    def decide(self, layer: int, expert: int, ctx: dict) -> bool:
        return True
