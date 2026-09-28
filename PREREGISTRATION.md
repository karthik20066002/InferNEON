# Preregistration — InferNEON routing and cache metrics

Locked 2026-09-28, BEFORE the 35-prompt drift results. Anything decided
after seeing those results is exploratory, not confirmatory. Ratifying
commit recorded at the bottom; the content above it is frozen there.

## Ground truths (two, never mixed)

- PRIMARY: FLA fast-kernel trace family, LM-logit hash `de6811…`
  (flash-linear-attention 0.5.2, fla-core 0.5.2, causal-conv1d 1.7.0).
  All future traces use this path.
- SECOND: reference-kernel trace family, hash `e4e302…`, kept solely for
  the dual-trace rule below. Never mix traces across paths in any analysis.

## Locked metric definitions (SET-based, rank-blind)

- Hit rate: a (prompt, token, layer) position is a HIT iff
  `routed_set ⊆ resident_set`. Hit rate = hits / positions. The resident
  set may be larger; ranking within either set is never consulted.
- Predictor accuracy: `predicted_set == routed_set` exactly, order ignored.
- Forbidden: top-1 accuracy, MRR, rank correlation, or any metric that
  rewards ranking within the 8. (Motivation, not data: the kernel-swap
  pilot showed most flips are rerankings of an unchanged set.)

## Noise floor and decision rule

- Let F = pooled fraction of (prompt, token, layer) positions where the
  routed expert SET differs between the two kernel paths, over all 35
  prompts (5 domains × 7: 3 calibration + 4 held-out), all prompt tokens +
  1 greedy decode token each, greedy, batch 1, seed 0, `--deterministic`,
  Qwen GPTQ-INT4, 40 layers.
- Pilot single-prompt value (motivation only, NOT the threshold): 82/920
  positions = 8.91% set-drift (7.01% of individual slots; 82/7360 = 1.11%
  per-slot expression of the same count).
- A policy beats another ONLY IF it wins on BOTH traces AND the margin
  exceeds F on BOTH. Anything smaller is noise.

## Latency (for the 128/512-tokens-per-prompt decision)

- Median decode seconds/token over ≥24 greedy tokens, cold page cache
  (`drop_caches` before each config), reported per cache mode with prefill
  stated separately. Single-token timings are not citable.

## Scope lock

- top-8 of 256 experts, 40 layers, greedy, batch 1, seed 0, text-only,
  `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` @ `00a66983516f8f8057741221277eed4141c9e431`.

Ratified in: 715c05bfa32ed3d824018a827c9517179520856a (content frozen here;
this line was the only change after it — verify with `git show`)
