#!/usr/bin/env python3
"""run_host_ref.py — host ground truth for the extended verification run.

Runs the real transformers OlmoeForCausalLM on N prompts x M decode tokens and
dumps, per step, the top-10 (id, logit) pairs. The firmware dumps the same
format over UART; tools/compare_run.py diffs them into a max|Δlogit| number.

Prompts are truncated to --max-prompt tokens because the firmware KV cache is
sized for MAX_TOKENS=128 total (prompt + decode). A truncated prefix is still
a real prefix of the real prompt.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = '/home/axiss/Documents/projects/InferNEON'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', default='summarization,tamil_hindi,math')
    ap.add_argument('--max-prompt', type=int, default=96)
    ap.add_argument('--new-tokens', type=int, default=32)
    ap.add_argument('--out', default=f'{ROOT}/inferneon-eval/reports/host_ref_extended.json')
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f'{ROOT}/models/olmoe-1b-7b/tokenizer.json')

    prompts = []
    for name in args.prompts.split(','):
        with open(f'{ROOT}/inferneon-eval/data/prompts/{name}.jsonl') as f:
            row = json.loads(f.readline())
        ids = tok.encode(row['text']).ids[:args.max_prompt]
        prompts.append({'prompt_id': row['prompt_id'], 'domain': row['domain'],
                        'n_prompt': len(ids), 'prompt_ids': ids})
        print(f"{row['prompt_id']:16s} {row['domain']:12s} {len(ids)} prompt tokens", flush=True)

    all_results = []
    for p in prompts:
        out = f"{ROOT}/inferneon-eval/reports/host_ref_{p['prompt_id']}.json"
        cmd = [f'{ROOT}/inferneon-eval/.venv/bin/python',
               f'{ROOT}/tools/olmoe_host_ref.py',
               '--model-dir', f'{ROOT}/models/olmoe-1b-7b',
               '--prompt-ids', ','.join(str(i) for i in p['prompt_ids']),
               '--new-tokens', str(args.new_tokens),
               '--out', out]
        print(f"--- running {p['prompt_id']} ({args.new_tokens} tokens)", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout[-1500:])
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-3000:])
            sys.exit(r.returncode)
        d = json.load(open(out))
        all_results.append({
            'prompt_id': p['prompt_id'], 'domain': p['domain'],
            'n_prompt': p['n_prompt'], 'prompt_ids': p['prompt_ids'],
            'generated_ids': d['generated_ids'], 'steps': d['steps'],
        })

    with open(args.out, 'w') as f:
        json.dump({'prompts': all_results}, f)
    print(f'wrote {args.out}', flush=True)
    for r in all_results:
        print(f"  {r['prompt_id']:16s} -> {r['generated_ids'][:8]}{'...' if len(r['generated_ids'])>8 else ''}", flush=True)


if __name__ == '__main__':
    main()
