#!/usr/bin/env python3
"""Proper decode-throughput timing: N tokens, median per token, cold page cache.

Single-token timings are dominated by disk reads (~20 GB checkpoint on
~14 GB RAM): each run finds a different fraction of weights in the page
cache, so per-run seconds swing wildly. This script drops caches before the
timed pass so every config starts cold, then reports median decode seconds
per token over --tokens decode steps. Run once per --mode; the medians are
what decide whether 128 or 512 tokens per prompt is realistic.

Needs a cached sudo timestamp for drop_caches: run `sudo -v` first, or pass
--no-drop-caches to skip (records page_cache_dropped=false; numbers from
such runs are not comparable to cold runs).
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from capture.stream_capture import StreamCapture
from capture.weight_cache import WeightCache
from capture.provenance import (package_versions, delta_kernel_path,
                                require_clean_tree, file_hashes)
from benchmark_weight_cache import record


def drop_caches():
    subprocess.run(['sync'], check=True)
    r = subprocess.run('echo 3 | sudo -n tee /proc/sys/vm/drop_caches',
                       shell=True, capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip() != '3':
        raise SystemExit(
            'drop_caches failed (need a cached sudo timestamp: run `sudo -v` '
            'in a terminal first, then relaunch), or pass --no-drop-caches.\n'
            f'stdout={r.stdout.strip()!r} stderr={r.stderr.strip()!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('off', 'cold', 'warm'), required=True)
    ap.add_argument('--tokens', type=int, default=24)
    ap.add_argument('--cache-mib', type=int, default=512)
    ap.add_argument('--no-drop-caches', action='store_true')
    ap.add_argument('--allow-dirty', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    gitstate = require_clean_tree(ROOT, args.allow_dirty, 'time_decode')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = ROOT / 'models/qwen-gptq-int4'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    text = 'Explain why the sky is blue in one short paragraph:'
    ids = tok.apply_chat_template([{'role': 'user', 'content': text}],
                                  add_generation_prompt=True, return_tensors='pt').input_ids[0].tolist()
    sc = StreamCapture(str(model))
    sc.weight_cache = WeightCache(0 if args.mode == 'off'
                                  else args.cache_mib * 1024 ** 2)

    def once():
        recs = [record() for _ in ids]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, logits = sc.prefill(ids, recs)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - t0
        toks, dts = [], []
        for i in range(args.tokens):
            token = int(logits.argmax())
            toks.append(token)
            rec = record()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, logits = sc.step(token, rec, pos=len(ids) + i)
            torch.cuda.synchronize()
            dts.append(time.perf_counter() - t0)
        return prefill, toks, dts

    page_cold = False
    with torch.inference_mode():
        if args.mode == 'warm':
            print('warm: untimed fill pass...', flush=True)
            once()  # populate the GPU weight cache; not timed
        if not args.no_drop_caches:
            print('dropping page cache...', flush=True)
            drop_caches()
            page_cold = True
        else:
            print('NOT dropping page cache (flag); numbers are hot-disk, not comparable.',
                  flush=True)
        prefill, toks, dts = once()
    out = {'label': 'MEASURED', 'mode': args.mode, 'assumed_test_parameters': vars(args),
           'prompt_tokens': len(ids), 'decode_tokens': args.tokens,
           'page_cache_dropped': page_cold,
           'delta_kernels': delta_kernel_path(),
           'versions': package_versions(),
           'git_commit': gitstate['git_commit'],
           'git_dirty': gitstate['git_dirty'],
           'git_tracked_dirty': gitstate['git_tracked_dirty'],
           'source_sha256': file_hashes(ROOT, ['capture/stream_capture.py',
                                               'scripts/time_decode.py']),
           'prefill_seconds': prefill,
           'decode_seconds_per_token': dts,
           'decode_median_seconds': statistics.median(dts),
           'decode_mean_seconds': statistics.mean(dts),
           'decode_min_seconds': min(dts),
           'decode_max_seconds': max(dts),
           'peak_vram_bytes': torch.cuda.max_memory_allocated(),
           'cache': sc.weight_cache.stats(),
           'output_text': tok.decode(toks, skip_special_tokens=True)}
    name = args.out or f'reports/time_decode_{args.mode}.json'
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    Path(name).write_text(json.dumps(out, indent=2) + '\n')
    print(f'{args.mode}: prefill={prefill:.1f}s median_decode={out["decode_median_seconds"]:.3f}s '
          f'min={min(dts):.3f}s max={max(dts):.3f}s page_cold={page_cold}', flush=True)


if __name__ == '__main__':
    main()
