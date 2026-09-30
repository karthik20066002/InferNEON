#!/usr/bin/env python3
"""pack_olmoe_full.py — append non-expert tensors + directory to the blob.

Input: models/olmoe-1b-7b safetensors + existing disk-olmoe.img (experts).
Output: disk-olmoe-full.img = copy + appended region, with a directory at
FIXED LBA 1 (512 B: magic2, count, entries of tag/offset/len/dtype).

Directory entries (tag, all little-endian):
  1 embed [50304,2048] raw bytes (bf16)
  2 routers, 16 x [64,2048] concatenated (bf16)
  3 norms, 33 x [2048] concatenated: per-layer input+post + final (as stored)
  4 attn, per layer q,k,v,o concatenated [4*2048,2048] (bf16)
  5 head [50304,2048] (bf16)
  6 rope_cos [MAXPOS,64] float32, host-exact (theta 10000, head_dim 128)
  7 rope_sin [MAXPOS,64] float32
dtype flag: 0 = bf16, 1 = fp32 (norms travel as stored; kernel converts).
MAXPOS = 128  # must match firmware config::MAX_TOKENS (KV + RoPE both sized by it)
"""
import argparse
import json
import math
import os
import shutil
import struct
import sys

import torch

DIR_MAGIC = 0x0000AE87
# The directory is NOT at a fixed LBA: the v1 expert blob starts at byte 512
# (base0 = round_up(16 + 16*8, 512)), so LBA 1 is expert (0,0)'s first 512
# bytes. Writing there silently corrupts one expert (boot log showed
# "expert00 head: 0x0000ae87"). The directory therefore lives immediately
# after the expert region, whose end the kernel recomputes from the LBA-0
# table: dir_lba = base0/512 + layers*experts*stride/512. See
# config::DIR_LBA_AFTER_EXPERTS.
MAXPOS = 128
HEAD_DIM = 128
ROPE_THETA = 10000.0


def read_shard_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n))


def walk_tensors(model_dir):
    """name -> (shape, dtype_str, file, data_begin, data_end)."""
    index = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))
    wmap = index['weight_map']
    headers = {}
    found = {}
    for name, fn in wmap.items():
        if name == '__metadata__':
            continue
        p = os.path.join(model_dir, fn)
        if fn not in headers:
            meta = read_shard_header(p)
            dbo = 8 + struct.unpack('<Q', open(p, 'rb').read(8))[0]
            headers[fn] = (meta, dbo)
        meta, dbo = headers[fn]
        info = meta[name]
        found[name] = (info['shape'], info['dtype'], fn,
                       dbo + info['data_offsets'][0], dbo + info['data_offsets'][1])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='models/olmoe-1b-7b')
    ap.add_argument('--in-img', default='disk-olmoe.img')
    ap.add_argument('--out-img', default='disk-olmoe-full.img')
    args = ap.parse_args()

    found = walk_tensors(args.model_dir)

    def get(base):
        shape, dtype, fn, b, e = found[base + '.weight']
        return shape, dtype, fn, b, e

    # Verify expected shapes (OLMoE-1B-7B) before writing anything.
    es, ed, _, _, _ = get('model.embed_tokens')
    assert es == [50304, 2048], es
    layers = 16
    blobs = {}
    blobs[1] = [('model.embed_tokens', es, ed)]
    rnames, nshapes, rkeys = [], [], []
    for li in range(layers):
        p = f'model.layers.{li}.'
        for suffix in ('input_layernorm', 'post_attention_layernorm'):
            s, d, _, _, _ = get(p + suffix)
            assert s == [2048], (suffix, s)
            nshapes.append((p + suffix, s, d))
        for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            s, d, _, _, _ = get(p + f'self_attn.{proj}')
            assert s == [2048, 2048], (proj, s)
        s, d, _, _, _ = get(p + 'mlp.gate')
        assert s == [64, 2048], s
        rnames.append(p + 'mlp.gate')
    s, d, _, _, _ = get('model.norm')
    assert s == [2048], s
    nshapes.append(('model.norm', s, d))
    qknames = []
    for li in range(layers):
        for nm in ('q_norm', 'k_norm'):
            base = f'model.layers.{li}.self_attn.{nm}'
            s, d, _, _, _ = get(base)
            assert s == [2048], (base, s)
            qknames.append(base)
    hs, hd, _, _, _ = get('lm_head')
    assert hs == [50304, 2048], hs
    print(f'embed {es} {ed}; norms {len(nshapes)} tensors; head {hs} {hd}',
          flush=True)

    # v2 = verbatim copy of v1 (LBA-0 table + expert bytes bit-identical) plus
    # appended tensors. Only the APPENDED part is new, so v1 images keep
    # booting to the storage probe unchanged.
    if os.path.exists(args.out_img) and os.path.getsize(args.out_img) > os.path.getsize(args.in_img):
        os.remove(args.out_img)
    shutil.copyfile(args.in_img, args.out_img)
    # NOTE: v2 layout keeps LBA0 + expert bytes IDENTICAL to the v1 image
    # (only appends past the expert end), so this stays a plain copy.
    # The directory lives AFTER the expert region at a table-computed
    # offset (see below) -- never at LBA1, which is expert (0,0)'s head.
    off = os.path.getsize(args.out_img)
    v1_size = off
    assert off % 512 == 0, off
    # Reserve the directory's own 512-byte block FIRST, so no tensor can be
    # emitted onto it (embed would otherwise start at exactly v1_size and the
    # directory write would silently eat its first 512 bytes = 128 bf16 values).
    dir_lba = v1_size // 512
    with open(args.out_img, 'ab') as f:
        f.write(b'\x00' * 512)
    off += 512
    print(f'directory reserved at LBA {dir_lba} (byte {v1_size}); tensors from {off}', flush=True)
    # Sanity: file must currently end exactly at the expert region end.
    index = json.load(open(os.path.join(args.model_dir, 'model.safetensors.index.json')))
    print(f'v1 expert region ends at {off}; directory goes there', flush=True)
    entries = []

    def emit(tag, data: bytes, dtype_flag: int):
        nonlocal off
        assert off % 512 == 0
        entries.append((tag, off, len(data), dtype_flag))
        with open(args.out_img, 'ab') as f:
            f.write(data)
            pad = (-len(data)) % 512
            if pad:
                f.write(b'\x00' * pad)
        off += len(data) + pad

    def raw_bytes(name):
        shape, dtype, fn, b, e = found[name + '.weight']
        with open(os.path.join(args.model_dir, fn), 'rb') as f:
            f.seek(b)
            data = f.read(e - b)
        assert len(data) == e - b
        return data, dtype

    # 1: embed
    data, dtype = raw_bytes('model.embed_tokens')
    emit(1, data, 0 if dtype == 'BF16' else 1)
    # 2: routers concatenated
    rdata = b''.join(raw_bytes(n)[0] for n in rnames)
    emit(2, rdata, 0)
    # 3: norms concatenated (as stored)
    ndata = b''.join(raw_bytes(n)[0] for n, _, _ in nshapes)
    nflag = 0
    emit(3, ndata, nflag)
    # 4: attn per layer q,k,v,o
    adata = b''
    for li in range(layers):
        for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            adata += raw_bytes(f'model.layers.{li}.self_attn.{proj}')[0]
    emit(4, adata, 0)
    # 5: head
    hdata, _ = raw_bytes('lm_head')
    emit(5, hdata, 0)
    # 6/7: rope tables, host-exact fp32
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    pos = torch.arange(MAXPOS).float()
    freqs = torch.outer(pos, inv)
    emit(6, freqs.cos().numpy().tobytes(), 1)
    emit(7, freqs.sin().numpy().tobytes(), 1)

    # Norm dtypes actually stored (informational, kernel branches on flag).
    print('norm dtypes:', sorted(set(d for _, _, d in nshapes)), flush=True)

    # tag 8: q/k norms
    qkdata = b''.join(raw_bytes(n)[0] for n in qknames)
    assert len(qkdata) == 32 * 2048 * 2
    emit(8, qkdata, 0)
    assert len(entries) == 8
    # Directory: the FIRST 512 bytes of the appended region, i.e. exactly at
    # the end of the (unmodified, verbatim-copied) v1 expert region. The
    # kernel finds it there by recomputing base0 + layers*experts*stride.
    # Every entry must start strictly after the directory block, and after the
    # expert region: a silent overlap here is the class of bug that made the
    # first boot print the directory magic as "expert00 head".
    for tag, o, ln, df in entries:
        assert o >= v1_size + 512, (tag, o, v1_size)
        assert (o - v1_size) % 512 == 0, (tag, o)
    assert off == v1_size + 512 + sum((ln + (-ln) % 512) for _, _, ln, _ in entries), (off, v1_size)
    block = struct.pack('<II', DIR_MAGIC, len(entries))
    for tag, o, ln, df in entries:
        block += struct.pack('<IQQI4x', tag, o, ln, df)
    assert len(block) <= 512, len(block)
    block += b'\x00' * (512 - len(block))
    with open(args.out_img, 'r+b') as f:
        f.seek(dir_lba * 512)
        f.write(block)
    print(f'directory at LBA {dir_lba} (byte {v1_size})', flush=True)
    print(f'wrote {args.out_img}: {os.path.getsize(args.out_img)} bytes, '
          f'{len(entries)} dir entries', flush=True)
    for tag, o, ln, df in entries:
        print(f'  tag {tag}: offset {o} len {ln} dtypeflag {df}', flush=True)


if __name__ == '__main__':
    if 'torch' not in sys.modules:
        pass
    main()
