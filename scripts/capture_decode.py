#!/usr/bin/env python3
"""Long decode capture with crash-safe checkpointing (the launch run).

Per prompt, writes data/traces/<prompt_id>.jsonl: a header line, then one
line per token (prompt tokens, then greedy decode tokens), each carrying the
token id plus per-layer top-8 ids and gate scores. Uncached weights
(WeightCache(0)): routes are bit-proven identical to cached runs, so traces
stay independent of cache state. Deterministic algorithms are always on:
resume-correctness depends on greedy decode reproducing tokens exactly.

Checkpointing: every token line is flushed + fsynced. Resume replays from
scratch (prefill, then decode) and VERIFIES each recomputed token id against
the recorded one before continuing; any mismatch aborts loudly instead of
forking the trace. A crash therefore loses at most the in-flight token, and
completed prompts are skipped by byte-exact line count.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from transformers import AutoTokenizer
from capture.stream_capture import StreamCapture
from capture.weight_cache import WeightCache
from capture.provenance import (package_versions, delta_kernel_path,
                                require_clean_tree, file_hashes)
from benchmark_weight_cache import record


def load_prompt(prompts_file, prompt_id):
    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get('prompt_id') == prompt_id:
                return d
    raise SystemExit(f'prompt {prompt_id!r} not in {prompts_file}')


def write_line(f, obj):
    f.write(json.dumps(obj) + '\n')
    f.flush()
    os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts-file', required=True)
    ap.add_argument('--prompt-id', required=True)
    ap.add_argument('--decode-tokens', type=int, default=128)
    ap.add_argument('--outdir', default='data/traces')
    ap.add_argument('--allow-dirty', action='store_true')
    args = ap.parse_args()
    gitstate = require_clean_tree(ROOT, args.allow_dirty, 'capture_decode')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG', '') not in (':4096:8', ':16:8'):
        print('WARNING: no CUBLAS_WORKSPACE_CONFIG=:4096:8; greedy resume '
              'needs deterministic reductions to reproduce tokens.', flush=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    wanted = load_prompt(args.prompts_file, args.prompt_id)
    text, domain = wanted['text'], wanted.get('domain')
    model = ROOT / 'models/qwen-gptq-int4'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = tok.apply_chat_template([{'role': 'user', 'content': text}],
                                  add_generation_prompt=True, return_tensors='pt').input_ids[0].tolist()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{args.prompt_id}.jsonl'

    def pack(rec, phase, index, token):
        return {'t': index, 'phase': phase, 'token': token,
                'top8': [t.tolist() for t in rec['top8']],
                'scores8': [[round(float(v), 6) for v in s] for s in rec['scores8']]}

    have = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    have.append(json.loads(line))
        head = have[0]
        assert head.get('type') == 'header' and head.get('prompt_id') == args.prompt_id, \
            f'{path}: header mismatch, refusing to append'
        assert head.get('text') == text, f'{path}: prompt text changed, refusing to append'
        assert head.get('decode_target') == args.decode_tokens, \
            f'{path}: decode_target changed ({head.get("decode_target")} vs {args.decode_tokens})'
    else:
        head = None
    n_done = len(have) - (1 if have else 0)
    total = len(ids) + args.decode_tokens
    if n_done >= total:
        print(f'{args.prompt_id}: COMPLETE ({n_done}/{total} tokens), skipping', flush=True)
        return
    print(f'{args.prompt_id}: resuming at token {n_done}/{total} '
          f'({len(ids)} prompt + {args.decode_tokens} decode)', flush=True)

    sc = StreamCapture(str(model))
    # 2 GiB GPU expert cache (routes bit-proven identical to uncached) plus
    # the RAM dequant memo: prefill pays each dequant once, decode reuses.
    # Peak VRAM stays under ~6.5 GiB on the 8 GiB card (measured 4.9 used).
    sc.weight_cache = WeightCache(2048 * 1024**2)
    mode = 'a' if have else 'w'
    with torch.inference_mode(), open(path, mode) as f:
        if head is None:
            head = {'type': 'header', 'prompt_id': args.prompt_id, 'domain': domain,
                    'text': text, 'prompt_tokens': len(ids),
                    'decode_target': args.decode_tokens,
                    'greedy': True, 'seed': 0, 'weight_cache_mib': 2048,
                    'deterministic_algorithms': True,
                    'delta_kernels': delta_kernel_path(),
                    'versions': package_versions(),
                    'git_commit': gitstate['git_commit'],
                    'git_tracked_dirty': gitstate['git_tracked_dirty'],
                    'source_sha256': file_hashes(ROOT, ['capture/stream_capture.py',
                                                        'scripts/capture_decode.py'])}
            write_line(f, head)
        recs = [record() for _ in ids]
        _, logits = sc.prefill(ids, recs)
        print(f'{args.prompt_id}: prefill {sc.phase_times["load"]:.1f}s load / '
              f'{sc.phase_times["drop"]:.1f}s drop', flush=True)
        for ti, rec in enumerate(recs):
            if ti < n_done:
                continue  # prefill recompute; record already on disk
            write_line(f, pack(rec, 'prefill', ti, ids[ti]))
        n_dec_done = max(0, n_done - len(ids))
        token = int(logits.argmax())  # decode token 0 comes from prefill logits
        for i in range(args.decode_tokens):
            if i < n_dec_done:
                # Replay to rebuild KV/recurrent state; verify the trace.
                want = have[1 + len(ids) + i]['token']
                if token != want:
                    raise SystemExit(
                        f'{args.prompt_id}: RESUME DIVERGENCE at decode {i}: '
                        f'recomputed {token} != recorded {want}. Determinism '
                        f'broken; trace left untouched.')
                _, logits = sc.step(token, None, pos=len(ids) + i)
            else:
                rec = record()
                t_tok = time.perf_counter()
                _, logits = sc.step(token, rec, pos=len(ids) + i)
                dt_tok = time.perf_counter() - t_tok
                write_line(f, pack(rec, 'decode', len(ids) + i, token))
                print(f'  {args.prompt_id} decode {i + 1}/{args.decode_tokens} '
                      f'token={token}', flush=True)
                if (i + 1) % 16 == 0:
                    pt = sc.phase_times
                    print(f'  phases last-token: {dt_tok:.2f}s wall = '
                          f'{pt["load"]:.2f}s load + {pt["drop"]:.2f}s drop + '
                          f'{dt_tok - pt["load"] - pt["drop"]:.2f}s fwd/other',
                          flush=True)
            token = int(logits.argmax())
        print(f'{args.prompt_id}: COMPLETE '
              f'({len(ids) + args.decode_tokens} tokens)', flush=True)


if __name__ == '__main__':
    main()
