"""sim.engine — event-driven replay stub (Phase C prep, no torch needed)."""
from dataclasses import dataclass, field

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
