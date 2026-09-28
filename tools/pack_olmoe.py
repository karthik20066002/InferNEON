#!/usr/bin/env python3
"""pack_olmoe.py — export OLMoE experts into the flat blob (host-side).

Verifies the uniform-expert-size invariant from the REAL safetensors headers,
computes expert_stride = round_up(expert_bytes, block_size), and emits the
LBA-0 layer table (magic 0xAE86). Full 12 GiB weight repack is a streamed
gather gated behind a future --write-blob flag (needs ~13 GiB free).

Usage:
    python3 tools/pack_olmoe.py --model-dir models/olmoe-1b-7b --block-size 512
    python3 tools/pack_olmoe.py --model-dir models/olmoe-1b-7b --write-table layer_table.bin
"""
from __future__ import annotations

import argparse
import json
import os
import struct

MAGIC_U32 = 0x0000AE86
# projection -> (expected shape, dtype bytes)
PROJECTIONS = {
    "gate_proj": ((1024, 2048), 2),
    "up_proj": ((1024, 2048), 2),
    "down_proj": ((2048, 1024), 2),
}


def round_up(value: int, align: int) -> int:
    assert align > 0 and (align & (align - 1)) == 0, "block size must be power of two"
    return (value + align - 1) // align * align


def read_shard_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def walk_experts(model_dir: str) -> dict:
    """Map (layer, expert, proj) -> (shape, dtype, file, begin, end).

    Aborts on ANY variance from the uniform-expert invariant — pure offset
    arithmetic is only valid if every expert is byte-identical in size.
    """
    found: dict = {}
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        hdr = read_shard_header(os.path.join(model_dir, fn))
        for name, meta in hdr.items():
            if name == "__metadata__" or ".mlp.experts." not in name:
                continue
            # model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight
            parts = name.split(".")
            layer = int(parts[2])
            expert = int(parts[5])
            proj = parts[6]
            assert proj in PROJECTIONS, f"unexpected projection {name}"
            exp_shape, exp_b = PROJECTIONS[proj]
            shape = tuple(meta["shape"])
            assert shape == exp_shape, f"SHAPE VARIANCE: {name} {shape} != {exp_shape}"
            assert meta["dtype"] == "BF16", f"DTYPE VARIANCE: {name} {meta['dtype']}"
            (begin, end) = meta["data_offsets"]
            n_elem = 1
            for d in shape:
                n_elem *= d
            assert end - begin == n_elem * exp_b, f"SIZE VARIANCE: {name}"
            found[(layer, expert, proj)] = (shape, meta["dtype"], fn, begin, end)
    return found


CHUNK = 4 * 1024 * 1024


def shard_data_start(model_dir: str, filename: str) -> int:
    """Byte offset where tensor payload begins (8-byte len + JSON header)."""
    with open(os.path.join(model_dir, filename), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    return 8 + n


def write_blob(args, layers: int, experts: int, stride: int) -> None:
    """Stream-gather every expert blob as gate||up||down into the flat image.

    Requires --model-dir (weights on disk). Memory-light: 4 MiB chunks.
    Expert blob e of layer l lands at base(l) + e*stride, where base(l) is
    the same LBA-0 table written by --write-table.
    """
    if not args.model_dir or not os.path.isdir(args.model_dir):
        raise SystemExit("--write-blob needs --model-dir with the safetensors shards")
    found = walk_experts(args.model_dir)
    assert len(found) == layers * experts * 3, f"tensor count {len(found)}"
    print(f"verified {len(found)} expert tensors: uniform, no variance")

    starts = {}
    for fn in sorted(os.listdir(args.model_dir)):
        if fn.endswith(".safetensors"):
            starts[fn] = shard_data_start(args.model_dir, fn)

    handles = {}

    def src_range(key):
        _shape, _dtype, fn, begin, end = found[key]
        if fn not in handles:
            handles[fn] = open(os.path.join(args.model_dir, fn), "rb")
        return handles[fn], starts[fn] + begin, end - begin

    def copy_range(fh_out, fh_in, src_off: int, dst_off: int, length: int) -> None:
        fh_in.seek(src_off)
        fh_out.seek(dst_off)
        remaining = length
        while remaining:
            step = CHUNK if remaining > CHUNK else remaining
            fh_out.write(fh_in.read(step))
            remaining -= step

    table_len = 16 + layers * 8
    base0 = round_up(table_len, args.block_size)
    total = base0 + layers * experts * stride
    # Blob packing assumes stride == 3 x projection bytes (no inter-blob pad).
    # walk_experts already proved every projection is exactly 4194304 B.
    assert stride == 3 * 4194304, f"stride {stride} breaks gate||up||down packing"
    print(f"writing {args.write_blob}: {total} bytes ({total / 2**30:.2f} GiB)")
    with open(args.write_blob, "wb") as out:
        out.truncate(total)
        out.seek(0)
        out.write(struct.pack("<IIII", MAGIC_U32, layers, stride, 0))
        for li in range(layers):
            out.write(struct.pack("<Q", base0 + li * experts * stride))
        for li in range(layers):
            layer_base = base0 + li * experts * stride
            for ei in range(experts):
                dst = layer_base + ei * stride
                for proj, sub in (("gate_proj", 0), ("up_proj", 1), ("down_proj", 2)):
                    fh_in, src_off, length = src_range((li, ei, proj))
                    copy_range(out, fh_in, src_off, dst + sub * (stride // 3), length)
            print(f"  layer {li:2d}/15 packed", flush=True)
    for fh in handles.values():
        fh.close()
    print(f"done: {args.write_blob}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/olmoe-1b-7b")
    ap.add_argument("--config", default=None, help="legacy: bare config.json path")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--write-table", default=None, help="write LBA-0 table to this path")
    ap.add_argument("--write-blob", default=None,
                    help="write the FULL flat image (LBA-0 table + 1024 expert "
                         "blobs as gate||up||down, 12 MiB each) to this path")
    args = ap.parse_args()

    if args.config and not os.path.isdir(args.model_dir):
        # Legacy fallback: config-only planning, no weight verification.
        with open(args.config) as f:
            cfg = json.load(f)
        hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
        layers, experts = cfg["num_hidden_layers"], cfg["num_experts"]
    else:
        with open(os.path.join(args.model_dir, "config.json")) as f:
            cfg = json.load(f)
        hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
        layers, experts = cfg["num_hidden_layers"], cfg["num_experts"]
        found = walk_experts(args.model_dir)
        # Expect exactly layers*experts*3 projection tensors, fully populated.
        assert len(found) == layers * experts * 3, f"tensor count {len(found)}"
        for li in range(layers):
            for ei in range(experts):
                for proj in PROJECTIONS:
                    assert (li, ei, proj) in found, f"missing L{li} E{ei} {proj}"
        print(f"verified {len(found)} expert tensors: uniform, no variance")

    per_expert_params = 3 * hidden * inter
    expert_bytes = per_expert_params * args.dtype_bytes
    stride = round_up(expert_bytes, args.block_size)
    assert stride % args.block_size == 0
    print(f"hidden={hidden} intermediate={inter} layers={layers} experts/layer={experts}")
    print(f"magic=0x{MAGIC_U32:08x} block={args.block_size}")
    print(f"expert_bytes={expert_bytes} stride={stride}")

    if args.write_table and not args.write_blob:
        # Header (16 B) + layers x u64 bases. Bases assume layers packed back to
        # back from the first block after the table; padding absorbed by entries.
        table_len = 16 + layers * 8
        base = round_up(table_len, args.block_size)
        with open(args.write_table, "wb") as f:
            f.write(struct.pack("<IIII", MAGIC_U32, layers, stride, 0))
            for li in range(layers):
                f.write(struct.pack("<Q", base + li * experts * stride))
        print(f"wrote {args.write_table} ({table_len} B table, first base {base})")
        return

    if args.write_blob:
        write_blob(args, layers, experts, stride)
        return

    print("TODO: pick --write-table (144 B header) or --write-blob (full image)")


if __name__ == "__main__":
    main()
