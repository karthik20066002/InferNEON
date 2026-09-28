#!/usr/bin/env python3
"""pack_qwen.py — export Qwen3.6-35B-A3B experts into the flat blob (host-side).

Numbers from the REAL upstream config (Qwen/Qwen3.6-35B-A3B text_config,
model_type qwen3_5_moe_text): hidden 2048, 40 layers, 256 routed experts of
moe_intermediate 512, top-8 routed + one always-on shared expert (512-wide
SwiGLU), silu. Tensor NAMES from model.safetensors.index.json.

Upstream layout deltas vs OLMoE (handled at export so the kernel is unchanged):
- routed gate+up are FUSED as `experts.gate_up_proj` [256, 1024, 2048];
  the packer SPLITS the fused rows back into gate/up blobs.
- one shared expert per layer (gate/up [512,2048], down [2048,512]) lives at
  a fixed offset after the routed region: shared_offset(layer) =
  layer_base + 256 * expert_stride.
- router `mlp.gate.weight` [256, 2048] + `shared_expert_gate` stay with the
  gating bundle (not in the expert blob region).

SCOPE: upstream BF16 layout. The eval team's GPTQ-INT4 artifact
(palmfuture/..., group 128) packs qweights differently — INT4 repack +
dequant kernels are an explicit TODO, not silently covered here.

Usage:
    python3 tools/pack_qwen.py --block-size 512
    python3 tools/pack_qwen.py --write-table layer_table_qwen.bin
    (add --model-dir <qwen weights> once pulled to walk + assert every tensor)
"""
from __future__ import annotations

import argparse
import json
import os
import struct

MAGIC_U32 = 0x0000AE86
HIDDEN = 2048
INTER = 512
LAYERS = 40
EXPERTS = 256
# upstream bf16 per routed/shared expert: 3*2048*512*2
EXPERT_BYTES = 3 * HIDDEN * INTER * 2

# name fragment -> (expected shape, dtype bytes). Fused gate_up is split at export.
ROUTED_TENSORS = {
    "mlp.experts.gate_up_proj": ((EXPERTS, 2 * INTER, HIDDEN), 2),
    "mlp.experts.down_proj": ((EXPERTS, HIDDEN, INTER), 2),
    "mlp.gate.weight": ((EXPERTS, HIDDEN), 2),
}
SHARED_TENSORS = {
    "mlp.shared_expert.gate_proj.weight": ((INTER, HIDDEN), 2),
    "mlp.shared_expert.up_proj.weight": ((INTER, HIDDEN), 2),
    "mlp.shared_expert.down_proj.weight": ((HIDDEN, INTER), 2),
}


def round_up(value: int, align: int) -> int:
    assert align > 0 and (align & (align - 1)) == 0, "block size must be power of two"
    return (value + align - 1) // align * align


def read_shard_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def walk_qwen(model_dir: str) -> dict:
    """Map (layer, key) -> header meta, asserting every expected shape.

    Aborts on ANY variance. Fused gate_up rows are verified splittable
    (first INTER rows = gate, next INTER = up) by shape only — the split
    itself happens in the future --write-blob repack.
    """
    expected = dict(ROUTED_TENSORS)
    expected.update(SHARED_TENSORS)
    found: dict = {}
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        hdr = read_shard_header(os.path.join(model_dir, fn))
        for name, meta in hdr.items():
            if name == "__metadata__" or ".mlp." not in name:
                continue
            frag = next((k for k in expected if k in name), None)
            if frag is None:
                continue
            exp_shape, exp_b = expected[frag]
            shape = tuple(meta["shape"])
            assert shape == exp_shape, f"SHAPE VARIANCE: {name} {shape} != {exp_shape}"
            parts = name.split(".")
            layer = int(parts[3])  # model.language_model.layers.{L}...
            found[(layer, frag)] = meta
    for li in range(LAYERS):
        for frag in expected:
            assert (li, frag) in found, f"missing L{li} {frag}"
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--write-table", default=None)
    args = ap.parse_args()

    if args.model_dir:
        found = walk_qwen(args.model_dir)
        print(f"verified {len(found)} qwen tensors: uniform, no variance")
    else:
        print("(no --model-dir: config-derived plan only, no weight walk)")

    stride = round_up(EXPERT_BYTES, args.block_size)
    shared_stride = round_up(EXPERT_BYTES, args.block_size)
    layer_region = EXPERTS * stride + shared_stride
    print(f"hidden={HIDDEN} inter={INTER} layers={LAYERS} routed/layer={EXPERTS} +1 shared")
    print(f"magic=0x{MAGIC_U32:08x} block={args.block_size}")
    print(f"expert_bytes={EXPERT_BYTES} stride={stride} layer_region={layer_region}")

    if args.write_table:
        table_len = 16 + LAYERS * 8
        base = round_up(table_len, args.block_size)
        with open(args.write_table, "wb") as f:
            f.write(struct.pack("<IIII", MAGIC_U32, LAYERS, stride, 0))
            for li in range(LAYERS):
                f.write(struct.pack("<Q", base + li * layer_region))
        print(f"wrote {args.write_table} ({table_len} B table, first base {base})")
        print("shared_offset(layer) = layer_base + 256*stride (documented, read by cache TODO)")
    else:
        print("TODO: --write-blob repack incl. fused gate_up split (needs weights on disk)")


if __name__ == "__main__":
    main()
