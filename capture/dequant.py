"""Dequantize Qwen3.6 GPTQ-INT4 (bits=4, group=128, sym) to fp16.

Formula verified against the installed GPTQModel loader and official
TorchLinear conversion (L7E33 gate_proj + spot checks, 2026-09-28):
  qweight [(in//8), out] int32, ROW-packed, 8 nibbles/low-first
  qzeros  [(in//128), (out//8)] int32, COL-packed
  scales  [(in//128), out] fp16
  g_idx identity (asserted per tensor)
  logical_zero = unpack(qzeros) + 1  # GPTQ v1 stores zero - 1
  W[i,j] = (q[i,j] - logical_zero[i//128,j]) * s[i//128,j]
The installed loader converts packed 0x77777777 qzeros to 0x88888888
before dequantization. The old `- 8` adjustment was wrong and created
large expert outputs followed by fp16 overflow.
 Numerics note: capture math runs on these fp16 dequants, NOT Marlin
 kernels. ULP-level divergence from a Marlin run is possible; hook
 recompute (§4.4) is self-consistent regardless. Recorded limitation.
"""
from __future__ import annotations

import torch

BITS = 4
GROUP = 128
PACK = 8


def unpack_rows(x: torch.Tensor) -> torch.Tensor:
    """int32 [r, c] -> uint [r*8, c], low nibble first."""
    x = x.to(torch.int64)
    out = torch.empty(x.shape[0] * PACK, x.shape[1], dtype=torch.int64)
    for k in range(PACK):
        out[k::PACK] = (x >> (4 * k)) & 0xF
    return out


def unpack_cols(x: torch.Tensor) -> torch.Tensor:
    """int32 [r, c] -> uint [r, c*8], low nibble first."""
    x = x.to(torch.int64)
    out = torch.empty(x.shape[0], x.shape[1] * PACK, dtype=torch.int64)
    for k in range(PACK):
        out[:, k::PACK] = (x >> (4 * k)) & 0xF
    return out


def dequant_linear(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize one [in, out] linear to out_dtype. Asserts g_idx identity."""
    in_f, out_f = qweight.shape[0] * PACK, qweight.shape[1]
    assert bool(((g_idx == torch.arange(in_f) // GROUP).all())), "non-identity g_idx"
    q = unpack_rows(qweight).to(torch.float64)
    # GPTQ v1 stores logical_zero - 1 on disk; match the loader's v1->v2
    # conversion before dequantization.
    z = unpack_cols(qzeros).to(torch.float64) + 1.0
    s = scales.to(torch.float64)
    g = torch.arange(in_f) // GROUP
    return ((q - z[g]) * s[g]).to(out_dtype)


def roundtrip_ok(qweight, qzeros, scales, g_idx) -> bool:
    """Bit-exact reproof of the stored integers from the dequant formula.

    Positions with scale == 0 are vacuously consistent: W is exactly 0
    there regardless of the stored nibble (dead groups carry arbitrary q),
    and 0/0 is undefined, so they are masked out of the check.
    """
    in_f = qweight.shape[0] * PACK
    q = unpack_rows(qweight)
    z = unpack_cols(qzeros).to(torch.float64) + 1.0
    s = scales.to(torch.float64)
    g = torch.arange(in_f) // GROUP
    w = (q.to(torch.float64) - z[g]) * s[g]
    live = s[g] != 0
    rt = torch.round(w / s[g].clamp_min(1e-300) + z[g]).to(torch.int64)
    return bool(((rt == q) | ~live).all())
