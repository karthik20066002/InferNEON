#!/usr/bin/env python3
"""Dump per-(token, layer) routes for kernel-sensitivity comparison.

Runs the standard prompt uncached and saves top-8 ids + scores for every
(token, layer) pair, plus LM-logit hashes and full provenance. Run once per
DeltaNet kernel path (fast vs --force-reference-kernels, both with
--deterministic), then diff with scripts/diff_routes.py. Never mix traces
across kernel paths.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
if '--force-reference-kernels' in sys.argv:
    from capture.provenance import block_fast_kernels
    block_fast_kernels()
import torch
from transformers import AutoTokenizer
from capture.stream_capture import StreamCapture
from capture.weight_cache import WeightCache
from capture.provenance import (package_versions, delta_kernel_path,
                                require_clean_tree, file_hashes)
from benchmark_weight_cache import run, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=1)
    ap.add_argument('--prompts-file', default=None,
                    help='JSONL with prompt_id/text fields; select with --prompt-id.')
    ap.add_argument('--prompt-id', default=None)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--allow-dirty', action='store_true')
    ap.add_argument('--force-reference-kernels', action='store_true')
    ap.add_argument('--out', default='reports/routes_dump.json')
    args = ap.parse_args()
    import os
    gitstate = require_clean_tree(ROOT, args.allow_dirty, 'dump_routes')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.deterministic:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG', '') not in (':4096:8', ':16:8'):
            print('WARNING: --deterministic without CUBLAS_WORKSPACE_CONFIG=:4096:8',
                  flush=True)
        torch.use_deterministic_algorithms(True)
    model = ROOT / 'models/qwen-gptq-int4'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if args.prompts_file:
        wanted = None
        with open(args.prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get('prompt_id') == args.prompt_id:
                    wanted = d
                    break
        if wanted is None:
            raise SystemExit(f'prompt {args.prompt_id!r} not in {args.prompts_file}')
        text, prompt_id = wanted['text'], wanted['prompt_id']
        domain = wanted.get('domain')
    else:
        text = 'Explain why the sky is blue in one short paragraph:'
        prompt_id = 'sky_pilot'
        domain = None
    ids = tok.apply_chat_template([{'role': 'user', 'content': text}],
                                  add_generation_prompt=True, return_tensors='pt').input_ids[0].tolist()
    sc = StreamCapture(str(model))
    sc.weight_cache = WeightCache(0)
    with torch.inference_mode():
        result = run(sc, ids, args.tokens)
    records = [{'top8': [t.tolist() for t in r['top8']],
                'scores8': [s.tolist() for s in r['scores8']]}
               for r in result['records']]
    out = {'label': 'MEASURED', 'prompt_id': prompt_id, 'domain': domain,
           'prompt_tokens': len(ids),
           'assumed_test_parameters': vars(args),
           'deterministic_algorithms': bool(args.deterministic),
           'delta_kernels': ('reference (forced)' if args.force_reference_kernels
                             else delta_kernel_path()),
           'versions': package_versions(),
           'git_commit': gitstate['git_commit'],
           'git_dirty': gitstate['git_dirty'],
           'git_tracked_dirty': gitstate['git_tracked_dirty'],
           'source_sha256': file_hashes(ROOT, ['capture/stream_capture.py',
                                               'scripts/dump_routes.py']),
           'logits_sha256': digest(result['logits']),
           'tokens': result['tokens'],
           'records': records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out) + '\n')
    print(f'wrote {args.out}: {len(records)} records x {len(records[0]["top8"])} layers, '
          f'logits {out["logits_sha256"][:12]}, kernels={out["delta_kernels"]}', flush=True)


if __name__ == '__main__':
    main()
