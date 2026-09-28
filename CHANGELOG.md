# CHANGELOG — decisions and lock events (newest first)

## 2026-09-28 — capture-only weight cache tested (MEASURED)
- Added `capture/weight_cache.py`: layer-qualified `(layer, expert)` keys,
  owned immutable tensors, fixed byte budget, no eviction policy. Prefill does
  not populate it; it is only a decode-capture performance cache.
- Added `tests/test_weight_cache.py` (3/3 pass) and
  `scripts/benchmark_weight_cache.py`.
- A 512 MiB visible Kitty run completed without system/GPU crash. It compared
  uncached repeat, cold cache, and warm cache using the same prompt/token path.
- Natural uncached GPU MoE reduction produced 228 route mismatches between
  repeated runs. Cold and warm cache also produced 228; LM argmax was 2/2 in
  each comparison, cached tensor digests stayed unchanged, and all tensors
  remained finite. Dense deltas are retained in
  `reports/weight_cache_benchmark.json` as diagnostics, not relabeled as
  cache corruption.
- Initial 2 GiB cache attempt was stopped before completion; no crash was
  observed. The harness now uses 512 MiB by default for this 8 GiB GPU.

## 2026-09-28 — Qwen streaming smoke fixed (MEASURED)
- Root cause of the earlier layer-2 overflow: the custom GPTQ dequantizer used
  the wrong qzero convention. The checkpoint is GPTQ v1: packed disk qzeros are
  `logical_zero - 1` (`0x77777777`); GPTQModel converts them to logical zero 8
  before dequantization. The corrected formula is
  `weight = (unpacked_qweight - (unpacked_qzeros + 1)) * scale`.
- Independent official `TorchLinear` check on a real L0E0 down projection:
  qzero 7 → runtime qzero 8; custom and GPTQModel dequant tensors matched
  exactly (max absolute difference 0.0).
- `capture/stream_capture.py` now records true `h_postmix` before the MoE
  norm and separately records the normalized router input.
- `scripts/smoke_stream.py` now uses the Qwen chat template and does not feed
  the last prompt token twice before decode.
- Corrected smoke: 22 prompt tokens + 2 decode tokens, 40 layers, 960
  router pairs, all finite; router recompute match 960/960 = 1.000000;
  peak VRAM 3.59 GiB. This is a validation smoke, not a full trace capture.

## 2026-09-28 — Q1–Q9 resolved via questions tool (all recommended options)
- Q1: primary `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` pinned at rev
  `00a66983516f8f8057741221277eed4141c9e431`; fallback `btbtyler09/GPTQ-4bit`
  → `Qwen/Qwen3.6-35B-A3B-FP8`.
- Q2: capture defaults kept (3 calib / 4 heldout, 512-token min, greedy, batch 1, seed 0).
- Q3: reporting kept (median+mean, P50/P95 E4, max/P95 added delay, E2 windowed plots).
- Q4: torch 2.14.0+cu130 kept (MEASURED working, CUDA True).
- Q5: disk offload under /home allowed, cap 60G.
- Q6: policy modules as ABCs (not Protocols); stub names kept.
- Q7: E7 partial agreement check if feasible, else declare impossible.
- Q8: git repo initialised in `inferneon-eval/`.
- Q9: 1-prompt capture validation first (brief step 2), then decide on full capture.

## 2026-09-28 — environment re-verified post-restart (MEASURED)
- RTX 4070 Laptop 8GB, driver 615.71.09; torch 2.14.0+cu130 CUDA True;
  transformers 5.17.0; accelerate 1.15.0; gptqmodel installed; 109G free.
- OLMoE BF16 smoke: 1.1 tok/s offloaded, greedy self-echo loop noted (1B-class
  behaviour, not a weights fault).

## 2026-09-28 — Qwen load OOM x3, filed as Q10 (MEASURED, negative result)
- `palmfuture/...GPTQ-Int4` rev 00a6698, device_map auto, offload_state_dict on.
- Kernel OOM-killer each run; anon peaks 8274224 / 10875900 / 11764900 kB.
- Load shims used (no routing impact): optimum get_layers filter for the BF16
  [1,2048] shared_expert_gate (Marlin N=1 wall); kept dense as stored.
- OLMoE-BF16 control loads and generates fine on this box (1.1 tok/s).
