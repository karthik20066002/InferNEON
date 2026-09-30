"""sim.storage — NVMe timing model stub.
Rules from brief §5.1.5: shared bandwidth B (not one-at-a-time), Q_dev
outstanding max, tiles of tile_bytes, per-request t_req, device queue
non-cancellable (completion always counted), software queue above it
reorderable/cancellable with P0 exact > P3 speculative.
"""
from dataclasses import dataclass

@dataclass
class StorageConfig:
    B: float = 400e6
    t_req: float = 100e-6
    Q_dev: int = 32
    tile_bytes: int = 262144

def transfer_ms(nbytes: int, cfg: StorageConfig) -> float:
    """ASSUMED linear model: t_req + nbytes/B. Real queueing in engine."""
    return cfg.t_req * 1000.0 + (nbytes / cfg.B) * 1000.0
