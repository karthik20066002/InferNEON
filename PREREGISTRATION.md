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

## Amendment 2026-09-29 — units fix: noise measured in the metric itself

No policy results exist; this change is legitimate. Reason: F (12.84%)
counts per-SPOT set changes, while hit rate counts per-EXPERT presence.
Most changed spots swap 1 of 8 experts, so per-expert drift is ~F/8, and
requiring a >12.84-point hit-rate margin would be ~8x too strict — real
differences (e.g. LFU over LRU by 5 points) would be discarded as noise.

- Hit-rate comparisons: for each policy P, run it on BOTH traces and take
  the wobble W(P) = |hit_rate_ref(P) − hit_rate_FLA(P)|, in hit-rate units,
  no conversion. A beats B only if it wins on BOTH traces with margin >
  max(W(A), W(B)) on EACH trace.
- Exact-set predictor accuracy stays on F (12.84% pooled): per-spot metric,
  same units as F. Win on both traces by > F.
- Per-domain claims use per-domain noise (F_domain / W_domain); pooled
  claims use pooled noise. Never apply the worst domain's number globally.
- Mechanism probe (pre-registered analysis, not a claim): the router's
  8th-vs-9th full-softmax score gap per domain. Hypothesis: smaller gaps on
  low-resource scripts explain fragile routing (tamil_hindi: 7 prompts, so
  its 18.53% carries wide error bars regardless).
