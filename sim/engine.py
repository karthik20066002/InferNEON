"""sim.engine — event-driven replay stub (Phase C prep, no torch needed)."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReplayEvent:
    """One atomic demand: a (prompt, token, layer) position needs all 8
    routed experts at once.

    Locked by PREREGISTRATION amendment 4. A position must NEVER be replayed
    as 8 sequential per-expert accesses: sequential replay lets recency
    policies score phantom hits across the sub-accesses and flips LRU-vs-LFU
    rankings (August 2026 trace-evaluation paper). The simulator deducts one
    joint miss/hit per event against the resident set.
    """
    prompt_id: str
    token_index: int
    layer: int
    experts: frozenset

    def __post_init__(self):
        if len(self.experts) != 8:
            raise ValueError(
                f'fused event needs exactly 8 experts, got {len(self.experts)}')
        if any(e < 0 for e in self.experts):
            raise ValueError('expert ids must be nonnegative')


def events_from_routed(prompt_id: str, top8_per_token_layer) -> list:
    """Build one ReplayEvent per (token, layer) from routed top-8 lists.

    top8_per_token_layer[ti][li] is the 8 expert ids in router order (order
    kept for record, ignored for scoring). Anything that is not exactly one
    8-set per position is rejected, never silently split or padded: pass
    fragments and you get an error, not 8 sequential events.
    """
    events = []
    for ti, per_layer in enumerate(top8_per_token_layer):
        for li, ids in enumerate(per_layer):
            ids = list(ids)
            if len(ids) != 8:
                raise ValueError(
                    f'{prompt_id} token {ti} layer {li}: '
                    f'expected 8 experts, got {len(ids)}')
            events.append(ReplayEvent(prompt_id, ti, li, frozenset(ids)))
    return events

@dataclass
class SimConfig:
    GOPS_sustained: float = 10.0  # ASSUMED
    B: float = 400e6  # bytes/s ASSUMED
    t_req: float = 100e-6  # ASSUMED
    Q_dev: int = 32  # ASSUMED
    tile_bytes: int = 262144  # 256 KiB ASSUMED
    capacity_slots: int = 1000
    expert_bytes: int = 1572864  # 1.5 MiB INT4 ASSUMED (3*2048*512/2)
    seed: int = 0

@dataclass
class TokenResult:
    compute_ms: float = 0.0
    unhidden_io_ms: float = 0.0
    predictor_ms: float = 0.0
    lm_head_ms: float = 0.0
    other_ms: float = 0.0
    bytes_exact: int = 0
    bytes_spec: int = 0
    bytes_wasted: int = 0

def run_stub(trace_path: str, cfg: SimConfig) -> list[TokenResult]:
    """Trivial reference: no prediction, always admit. Real engine replaces this."""
    raise NotImplementedError("sim.engine real loop pending Phase C")
