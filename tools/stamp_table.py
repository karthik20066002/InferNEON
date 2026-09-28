#!/usr/bin/env python3
"""stamp_table.py — write the verified LBA-0 layer table onto the stub disk.

Uses the exact constants verified against the real OLMoE-1B-7B weights
(3072 tensors walked by pack_olmoe.py: uniform, no variance):
  magic 0xAE86, 16 layers, expert stride 12582912 (12 MiB, 512-aligned).
Regenerate byte-identically with:
  pack_olmoe.py --model-dir models/olmoe-1b-7b --write-table layer_table.bin
once the weights are back on disk. Until --write-blob lands, only LBA-0
carries real data; the rest of the stub image is zeros.

Usage:
    python3 tools/stamp_table.py disk.img [--block-size 512]
"""
from __future__ import annotations

import struct
import sys

MAGIC_U32 = 0x0000AE86
NUM_LAYERS = 16
NUM_EXPERTS = 64
EXPERT_STRIDE = 12582912

QWEN = {
    # Verified against Qwen/Qwen3.6-35B-A3B text_config (qwen3_5_moe_text).
    # Upstream bf16 per expert: 3*2048*512*2. Shared expert: same size,
    # one per layer after the routed region.
    "layers": 40,
    "experts": 256,
    "stride": 6291456,
    "shared_bytes": 6291456,
}


def table_bytes(layers, experts, stride, block, shared_bytes=0):
    table_len = 16 + layers * 8
    base = (table_len + block - 1) // block * block
    region = experts * stride + ((shared_bytes + block - 1) // block * block if shared_bytes else 0)
    table = struct.pack("<IIII", MAGIC_U32, layers, stride, 0)
    for li in range(layers):
        table += struct.pack("<Q", base + li * region)
    return table, base


def main() -> None:
    disk = sys.argv[1] if len(sys.argv) > 1 else "disk.img"
    model = "olmoe"
    block = 512
    argv = sys.argv[2:]
    while argv:
        a = argv.pop(0)
        if a == "--model":
            model = argv.pop(0)
        elif a == "--block-size":
            block = int(argv.pop(0))
    if model == "qwen":
        q = QWEN
        assert q["stride"] % block == 0, "stride/block mismatch"
        table, base = table_bytes(q["layers"], q["experts"], q["stride"], block, q["shared_bytes"])
        layers, stride = q["layers"], q["stride"]
    else:
        assert EXPERT_STRIDE % block == 0, "stride/block mismatch"
        table, base = table_bytes(NUM_LAYERS, NUM_EXPERTS, EXPERT_STRIDE, block)
        layers, stride = NUM_LAYERS, EXPERT_STRIDE
    with open(disk, "r+b") as f:
        f.seek(0)
        f.write(table)
    print(f"stamped LBA-0 of {disk} [{model}]: magic=0x{MAGIC_U32:08x} "
          f"layers={layers} stride={stride} base0={base}")


if __name__ == "__main__":
    main()
