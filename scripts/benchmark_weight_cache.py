#!/usr/bin/env python3
"""Cache corruption/timing regression: off, cold, then warm; same token inputs.

No held-out prompts and no simulator policies. Strict equality tests isolate
cache effects, not equivalence to another inference backend. Results include
measured timing and digests. Run from inferneon-eval with its venv Python.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from capture.stream_capture import StreamCapture
from capture.weight_cache import WeightCache

FIELDS = ('h_in', 'h_postmix', 'router_input', 'top8', 'scores8', 'logits256')


def record():
    return {key: [] for key in FIELDS}


def digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate(sc, records, batched):
    """Recompute the real router including FP32 softmax, in live batch shape."""
    for layer in range(sc.nlayers):
        groups = [records] if batched else [[r] for r in records]
        for group in groups:
            for r in group:
                for key in FIELDS:
                    if not torch.isfinite(r[key][layer]).all():
                        raise AssertionError(f'nonfinite {key}, layer {layer}')
                ids = r['top8'][layer].tolist()
                assert len(ids) == len(set(ids)) == sc.cfg.num_experts_per_tok
                assert all(0 <= i < sc.cfg.num_experts for i in ids)
            h = torch.stack([r['h_postmix'][layer] for r in group]).to(sc.device)
            lay = sc.model.model.layers[layer]
            x = lay.post_attention_layernorm(h)
            logits, scores, indices = lay.mlp.gate(x)
            for i, r in enumerate(group):
                torch.testing.assert_close(indices[i].cpu(), r['top8'][layer], rtol=0, atol=0)
                torch.testing.assert_close(scores[i].float().cpu(), r['scores8'][layer], rtol=0, atol=0)


def run(sc, ids, steps, forced=None):
    prompt = [record() for _ in ids]
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, logits = sc.prefill(ids, prompt)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    all_logits = [logits]
    decode = []
    times = []
    tokens = []
    for i in range(steps):
        token = int(logits.argmax()) if forced is None else forced[i]
        tokens.append(token)
        rec = record()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, logits = sc.step(token, rec, pos=len(ids) + i)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        decode.append(rec)
        all_logits.append(logits)
        print(f'  decode {i+1}/{steps}: {times[-1]:.3f}s token={token}', flush=True)
    for tensor in all_logits:
        assert torch.isfinite(tensor).all(), 'nonfinite LM logits'
    validate(sc, prompt, batched=True)
    validate(sc, decode, batched=False)
    return dict(prefill_seconds=prefill_seconds, decode_seconds=times,
                tokens=tokens, records=prompt + decode, logits=all_logits)


def compare(a, b):
    max_logit_delta = 0.0
    max_hidden_delta = 0.0
    max_router_delta = 0.0
    lm_argmax_same = 0
    for x, y in zip(a['logits'], b['logits'], strict=True):
        max_logit_delta = max(max_logit_delta, float((x - y).abs().max()))
        lm_argmax_same += int(int(x.argmax()) == int(y.argmax()))
    route_mismatches = []
    # Expert-overlap diagnostics: at each mismatched (token, layer) position,
    # how many of the 8 routed experts differ? Router `top8` is in score order,
    # so index 7 is the lowest-weight expert on a near-tie.
    expert_diff_histogram: dict[int, int] = {}
    eighth_only_mismatches = 0
    order_only_mismatches = 0
    per_layer_mismatch_counts: dict[int, int] = {}
    for rec_index, (x, y) in enumerate(zip(a['records'], b['records'], strict=True)):
        for layer in range(len(x['top8'])):
            xa = x['top8'][layer].tolist()
            yb = y['top8'][layer].tolist()
            # Routing IDs are the trace correctness bar. Natural GPU
            # nondeterminism is reported and compared against off1/off2.
            if xa != yb:
                # Canonical 2-tuple (rec_index, layer). Older files used a
                # 3-tuple with a prompt-label prefix; normalize on read.
                route_mismatches.append((rec_index, layer))
                per_layer_mismatch_counts[layer] = per_layer_mismatch_counts.get(layer, 0) + 1
                set_a, set_b = set(xa), set(yb)
                n_differ = 8 - len(set_a & set_b)
                expert_diff_histogram[n_differ] = expert_diff_histogram.get(n_differ, 0) + 1
                if n_differ == 0:
                    order_only_mismatches += 1
                if xa[:7] == yb[:7] and xa[7] != yb[7]:
                    eighth_only_mismatches += 1
            max_hidden_delta = max(max_hidden_delta,
                float((x['h_postmix'][layer] - y['h_postmix'][layer]).abs().max()))
            max_router_delta = max(max_router_delta,
                float((x['logits256'][layer] - y['logits256'][layer]).abs().max()))
    return dict(max_logit_delta=max_logit_delta,
                max_hidden_delta=max_hidden_delta,
                max_router_delta=max_router_delta,
                lm_argmax_same=lm_argmax_same,
                lm_argmax_total=len(a['logits']),
                route_mismatch_count=len(route_mismatches),
                route_mismatches=route_mismatches,
                expert_diff_histogram=expert_diff_histogram,
                eighth_only_mismatches=eighth_only_mismatches,
                order_only_mismatches=order_only_mismatches,
                per_layer_mismatch_counts=per_layer_mismatch_counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=3)
    ap.add_argument('--cache-mib', type=int, default=512)
    ap.add_argument('--out', default='reports/weight_cache_benchmark.json')
    ap.add_argument('--deterministic', action='store_true',
                    help='Enable torch.use_deterministic_algorithms(True) for the '
                         'uncached off1/off2 repeat. Requires CUBLAS_WORKSPACE_CONFIG '
                         '=:4096:8 (or :16:8) in the environment. The Qwen MoE combine '
                         'step (index_add_ with duplicate token indices) is the usual '
                         'nondeterminism source; deterministic mode serializes it. '
                         'Other kernels (DeltaNet reference fallback) may still throw '
                         'or vary; the run records whatever happens.')
    args = ap.parse_args()
    if args.tokens < 1 or args.cache_mib < 1:
        ap.error('tokens and cache-mib must be positive')
    import os
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = bool(args.deterministic)
    torch.backends.cudnn.benchmark = not bool(args.deterministic)
    cublas_cfg = os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')
    deterministic = bool(args.deterministic)
    if deterministic:
        if cublas_cfg not in (':4096:8', ':16:8'):
            print(f'WARNING: --deterministic without CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                  f'(got {cublas_cfg!r}); cublas reductions may stay nondeterministic '
                  f'or torch may throw. Export it before relaunching for a strict test.',
                  flush=True)
        # Hard fail on nondeterministic ops so off1-vs-off2 either matches
        # exactly or tells us which op refuses determinism.
        torch.use_deterministic_algorithms(True)
    else:
        # Explicitly record why off1 vs off2 is expected to vary: the MoE
        # combine (Qwen3_5MoeExperts.forward: final_hidden_states.index_add_
        # with duplicate token_idx) sums in atomic order on CUDA.
        pass
    model = ROOT / 'models/qwen-gptq-int4'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    text = 'Explain why the sky is blue in one short paragraph:'
    ids = tok.apply_chat_template([{'role': 'user', 'content': text}],
                                 add_generation_prompt=True, return_tensors='pt').input_ids[0].tolist()
    # Reuse the same skeleton; prefill resets attention state for each run.
    # Changing only the weight cache keeps all compute and tensor layouts fixed.
    sc = StreamCapture(str(model))
    output = {'label': 'MEASURED', 'prompt': text, 'prompt_tokens': len(ids),
              'assumed_test_parameters': vars(args), 'seed': 0,
              'deterministic_algorithms': deterministic,
              'versions': {k: importlib.metadata.version(k) for k in
                           ('torch', 'transformers', 'safetensors')},
              'checkpoint_config_sha256': hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT/'capture/stream_capture.py', ROOT/'capture/dequant.py',
                                          ROOT/'capture/weight_cache.py', Path(__file__))},
              'runs': {}}
    commit = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True)
    output['git_commit'] = commit.stdout.strip() if commit.returncode == 0 else None
    output['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
    output['cublas_workspace_config'] = cublas_cfg
    output['deterministic_algorithms_enabled'] = torch.are_deterministic_algorithms_enabled()
    output['timing_note'] = (
        'Single-run prefill/decode seconds are diagnostics only, not a cache '
        'speedup claim. Observed uncached variance alone (decode ~5.2s vs ~8.8s, '
        'prefill ~50-77s across runs) exceeds any weight-cache effect at this '
        'token count. Do not cite a cache speedup from this script.')
    baseline = None
    baseline_digest = None
    before = {}
    comparison = {}
    with torch.inference_mode():
        for mode in ('off1', 'off2', 'cold', 'warm'):
            if mode in ('off1', 'off2'):
                sc.weight_cache = WeightCache(0)
            elif mode == 'cold':
                sc.weight_cache = WeightCache(args.cache_mib * 1024**2)
            if mode == 'warm':
                before = {str(k): digest(v) for k, v in sc.weight_cache.entries.items()}
            torch.cuda.reset_peak_memory_stats()
            print(f'RUN {mode}', flush=True)
            result = run(sc, ids, args.tokens, forced=None if baseline is None else baseline['tokens'])
            result_digest = digest(result['logits'])
            if baseline is None:
                baseline = result
                baseline_digest = result_digest
                comparison = {}
            else:
                comparison = compare(baseline, result)
                print(f'{mode}: route_mismatches={comparison["route_mismatch_count"]} '
                      f'argmax={comparison["lm_argmax_same"]}/{comparison["lm_argmax_total"]} '
                      f'max_logit_delta={comparison["max_logit_delta"]:.4f} '
                      f'eighth_only={comparison["eighth_only_mismatches"]} '
                      f'expert_hist={comparison["expert_diff_histogram"]}',
                      flush=True)
            if mode == 'warm':
                after = {str(k): digest(v) for k, v in sc.weight_cache.entries.items()}
                assert all(after[k] == value for k, value in before.items()), 'cached tensor mutated'
                output['cached_tensor_digests_unchanged'] = True
                output['cached_entries_checked'] = len(before)
            stats = sc.weight_cache.stats()
            logits_equal_off1 = (result_digest == baseline_digest)
            routes_equal_off1 = (comparison == {} or comparison.get('route_mismatch_count') == 0)
            output['runs'][mode] = {k: result[k] for k in ('prefill_seconds', 'decode_seconds', 'tokens')}
            output['runs'][mode].update(cache=stats,
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                logits_sha256=result_digest,
                # Provenance label (not a comparison): which runs bypass the cache.
                is_uncached_repeat=(mode in ('off1', 'off2')),
                # Real exactness checks vs the off1 baseline. off1 itself is
                # trivially exact; off2/cold/warm are exact only with 0 route
                # mismatches AND identical LM-logit hashes. Previously this
                # field was hardcoded to (mode in off1/off2) and compared
                # nothing, while the log line printed "EXACT PASS"
                # unconditionally for every mode.
                logits_equal_off1=logits_equal_off1,
                routes_exactly_equal_off1=routes_equal_off1,
                exact_vs_uncached=(logits_equal_off1 and routes_equal_off1),
                router_pairs_validated=len(result['records']) * sc.nlayers)
            if mode not in ('off1',):
                output['runs'][mode]['comparison_vs_off1'] = comparison
            print(f'{mode}: prefill={result["prefill_seconds"]:.3f}s cache={stats} '
                  f'routes_exact={routes_equal_off1} logits_equal={logits_equal_off1} '
                  f'(timing is noise, not a speedup claim)', flush=True)
    def _norm_positions(entries):
        # Normalize legacy 3-tuples ("label", rec, layer) to 2-tuples
        # (rec, layer) so set comparisons work across old/new files.
        norm = set()
        for e in (entries or []):
            t = tuple(e)
            norm.add(t[-2:])
        return norm
    natural = _norm_positions(output['runs'].get('off2', {})
                  .get('comparison_vs_off1', {}).get('route_mismatches', []))
    cache_review = {}
    for mode in ('cold', 'warm'):
        observed = _norm_positions(output['runs'].get(mode, {})
                       .get('comparison_vs_off1', {}).get('route_mismatches', []))
        cache_review[mode] = {
            'natural_repeat_mismatches': sorted(natural),
            'observed_mismatches': sorted(observed),
            'new_mismatches_beyond_off_repeat': sorted(observed - natural),
        }
    output['cache_corruption_check'] = cache_review
    natural_cmp = output['runs'].get('off2', {}).get('comparison_vs_off1', {})
    natural_count = natural_cmp.get('route_mismatch_count', 0)
    # Deterministic baseline achieved iff two identical uncached runs agree
    # exactly (0 route mismatches + identical logit hashes). Only then does
    # "cold/warm == off1" become a yes/no proof instead of noise-vs-noise.
    deterministic_baseline = bool(
        natural_cmp.get('route_mismatch_count') == 0
        and output['runs'].get('off2', {}).get('logits_equal_off1') is True)
    output['deterministic_baseline_achieved'] = deterministic_baseline
    cache_envelope_ok = {}
    for mode in ('cold', 'warm'):
        cmp = output['runs'].get(mode, {}).get('comparison_vs_off1', {})
        cache_envelope_ok[mode] = {
            # Dense deltas and single-run timings are diagnostics only: one
            # uncached repeat already shows natural GPU reduction variation
            # (MoE index_add_ atomics; ~25% of the 920 route positions flip
            # between identical runs, drift growing with depth). The hard
            # checks are route mismatch count, token choice, finite tensors,
            # cache digests, and -- under a deterministic baseline -- exact
            # digest + position-set identity.
            'route_count_same_as_uncached_repeat': cmp.get('route_mismatch_count') == natural_count,
            'lm_argmax_same': cmp.get('lm_argmax_same') == cmp.get('lm_argmax_total'),
            'all_finite': True,
            'logits_equal_off1': output['runs'][mode].get('logits_equal_off1'),
            'routes_exactly_equal_off1': output['runs'][mode].get('routes_exactly_equal_off1'),
            'max_logit_delta': cmp.get('max_logit_delta'),
            'max_hidden_delta': cmp.get('max_hidden_delta'),
            'max_router_delta': cmp.get('max_router_delta'),
            'eighth_only_mismatches': cmp.get('eighth_only_mismatches'),
            'expert_diff_histogram': cmp.get('expert_diff_histogram'),
            'order_only_mismatches': cmp.get('order_only_mismatches'),
            'per_layer_mismatch_counts': cmp.get('per_layer_mismatch_counts'),
        }
    output['cache_corruption_check']['envelope'] = cache_envelope_ok
    output['cache_corruption_check']['position_sets_identical'] = {
        mode: _norm_positions(output['runs'][mode]['comparison_vs_off1']
                  .get('route_mismatches', [])) == natural
        for mode in ('cold', 'warm')
    }
    output['output_text'] = tok.decode(baseline['tokens'], skip_special_tokens=True)
    if deterministic_baseline:
        # Hard proof regime: cache must reproduce off1 bit-for-bit.
        cache_exact = all(
            output['runs'][m].get('exact_vs_uncached') is True
            for m in ('cold', 'warm'))
        if cache_exact:
            output['status'] = 'PASS'
        else:
            output['status'] = 'CORRUPTION_BUG'
    else:
        # Noise-vs-noise regime: equal counts prove nothing (the 228/228
        # match was a coincidence of flip-prone positions flipping at
        # random). Never promote to PASS; cap at REVIEW until determinism
        # is fixed and off1-vs-off2 drops to 0 mismatches.
        output['status'] = 'REVIEW'
        output['status_note'] = (
            'Uncached off1-vs-off2 disagree '
            f'({natural_count} mismatches, '
            f'max_logit_delta={natural_cmp.get("max_logit_delta")}); '
            'cache-vs-uncached is noise vs noise. Fix determinism first '
            '(--deterministic + CUBLAS_WORKSPACE_CONFIG=:4096:8), rerun until '
            'off1/off2 match exactly, then re-evaluate the cache.')
    # Exact position-set identity is the requested cache-inertness criterion;
    # an equal mismatch count is insufficient and must never be promoted to PASS.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2) + '\n')
    print(f'CORRUPTION CHECK: {output["status"]}; output:',
          repr(output['output_text']), flush=True)


if __name__ == '__main__':
    main()
