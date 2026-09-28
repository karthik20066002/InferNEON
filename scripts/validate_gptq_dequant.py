#!/usr/bin/env python3
"""Compare the streaming dequantizer with GPTQModel's official v1 loader.

This loads only isolated expert tensors, never the whole Qwen model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "qwen-gptq-int4"
sys.path.insert(0, str(ROOT / "capture"))
from dequant import dequant_linear  # noqa: E402


def check(base: str) -> None:
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = {suffix: index[base + "." + suffix] for suffix in
             ("qweight", "qzeros", "scales", "g_idx")}
    with safe_open(str(MODEL / names["qweight"]), framework="pt") as shard:
        raw = {
            suffix: shard.get_slice(base + "." + suffix)[:]
            for suffix in names
        }

    in_features = raw["g_idx"].numel()
    out_features = raw["scales"].shape[1]
    official = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
        format=FORMAT.GPTQ,
        backend=BACKEND.GPTQ_TORCH,
    )
    official.load_state_dict(raw, strict=True)
    convert_gptq_v1_to_v2_format_module(
        official, bits=4, pack_dtype=torch.int32
    )
    official.eval()
    expected = official._dequantize_weight_cached_248()
    actual = dequant_linear(*(
        raw[k] for k in ("qweight", "qzeros", "scales", "g_idx")
    ), out_dtype=expected.dtype)
    delta = (actual - expected).abs().max().item()
    print(f"{base}: qzero_runtime={int(official.qzeros.flatten()[0] & 0xf)} "
          f"max_abs_diff={delta:.8g}")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    check("model.language_model.layers.0.mlp.experts.0.down_proj")
    check("model.language_model.layers.7.mlp.experts.33.gate_proj")
    print("GPTQ DEQUANT VALIDATION: PASS")
