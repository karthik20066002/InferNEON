#!/usr/bin/env python3
"""compare_run.py — diff firmware UART output against host ground truth.

The firmware prints, per decode step, the top-10 (id, logit) pairs it computed
and its argmax token. The host JSON has the same. This produces the number a
reviewer actually wants:

    max |Δlogit| = X over N steps   (and the token match rate)

Two distinct questions are kept separate on purpose:
  * Did the argmax match?  (token-level correctness)
  * How close were the logits? (numerical fidelity -- a matched argmax with a
    5.0 logit error is a different claim than a 0.001 error)

The firmware log format (see src/infer.rs):
    head shard:0xN best:0xID      x8   (heartbeats, ignored here)
    lg <rank> <id:hex> <logit:f32hex>  x10
    tok: 0xID
"""
import argparse
import json
import re
import struct
import sys


def parse_firmware_log(path):
    """-> list of steps, each {'top10': [(id, logit), ...], 'tok': int}."""
    steps = []
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            m = re.match(r'^lg ([0-9a-f]+) ([0-9a-f]{8}) ([0-9a-f]{8})$', line)
            if m:
                if cur is None:
                    cur = {'top10': [], 'tok': None}
                rank = int(m.group(1), 16)
                tid = int(m.group(2), 16)
                bits = int(m.group(3), 16)
                logit = struct.unpack('<f', struct.pack('<I', bits))[0]
                cur['top10'].append((rank, tid, logit))
                continue
            m = re.match(r'^tok: ([0-9a-f]{8})$', line)
            if m:
                if cur is not None:
                    cur['tok'] = int(m.group(1), 16)
                    cur['top10'].sort()
                    steps.append(cur)
                    cur = None
                continue
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('firmware_log')
    ap.add_argument('host_json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    fw = parse_firmware_log(args.firmware_log)
    host = json.load(open(args.host_json))
    # host_json may be the multi-prompt file or a single-prompt file
    if 'prompts' in host:
        host_steps = []
        for p in host['prompts']:
            for s in p['steps']:
                host_steps.append(s)
        prompt_meta = [(p['prompt_id'], p['n_prompt']) for p in host['prompts']]
    else:
        host_steps = host['steps']
        prompt_meta = [(host.get('model_dir', 'single'), 0)]

    n = min(len(fw), len(host_steps))
    if len(fw) != len(host_steps):
        print(f'WARNING: step count mismatch firmware={len(fw)} host={len(host_steps)}', file=sys.stderr)

    token_match = 0
    max_dlogit = 0.0
    max_dlogit_at = None
    per_step = []
    for i in range(n):
        f, h = fw[i], host_steps[i]
        tok_ok = f['tok'] == h['next']
        token_match += tok_ok
        # host logits by id
        host_lut = {int(h['top10_ids'][k]): float(h['top10_logits'][k]) for k in range(10)}
        step_max = 0.0
        for _rank, tid, fl in f['top10']:
            if tid in host_lut:
                d = abs(fl - host_lut[tid])
                if d > step_max:
                    step_max = d
                if d > max_dlogit:
                    max_dlogit = d
                    max_dlogit_at = (i, tid, fl, host_lut[tid])
        per_step.append({'step': i, 'tok_ok': tok_ok, 'fw_tok': f['tok'],
                         'host_tok': h['next'], 'max_dlogit': round(step_max, 6)})

    result = {
        'n_steps': n,
        'token_match': token_match,
        'token_match_rate': round(token_match / n, 4) if n else 0.0,
        'max_abs_dlogit': round(max_dlogit, 6),
        'max_abs_dlogit_at': max_dlogit_at,
        'prompts': prompt_meta,
        'per_step': per_step,
    }
    print(json.dumps({k: v for k, v in result.items() if k != 'per_step'}, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
