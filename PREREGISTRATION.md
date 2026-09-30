# Preregistration — InferNEON routing and cache metrics

## Current scope — matched-L exposed wait (supersedes prior headlines)

Locked before the matched-L runs. Reason: unequal lookahead and per-hint
bandwidth arithmetic cannot establish whether hints help one storage lane.
The historical recall/E4 headlines below are suspended, not implemented.

- Only oracle: demand-time perfect IDs, `predict(j) = D_j`, implemented in
  `sim/predictors.py::PerfectIDs.predict`. Belady-eviction is not this oracle.
- Only current comparison: LRU/no hint, B1 (previous decode token's 8 at
  the same layer), perfect IDs; both hint policies at L = 0, 1, 4, 40
  layer events. Release target j at event j-L; no pre-start hints. B1 has
  no first-token history. At L=40, B1 uses the current routed event's IDs.
- Decode only, both families separately, the frozen 20 held-out prompts
  listed below. Empty cache, lane, and predictor history per prompt; no
  prefill warmup. Pool total wait / decode tokens within each family.
- Existing capacities: 320, 640, 1000, 2000, 3000, 4000, 6500 slots.
  Bandwidth 100e6 and 413e6 bytes/s is ASSUMED. Compute slack 8 ms/layer
  event is ASSUMED (the prior test-local value, not measured or fitted).
  Expert transfer size 1,572,864 bytes is ASSUMED INT4 payload, excluding
  metadata/request overhead. One serial lane, fixed LRU, no cancellation.
- Outcome: exposed wait, not recall. Descriptive simulated rankings only;
  no significance or hardware claims. Preserve zero/negative hint gains.
- Deliver `reports/exposed_wait_matched_L.json`, one exposed-wait plot,
  and a 10-line note, then stop. No E4 always-wrong run, cap, cancel,
  worst-case bound, new predictor, hardware work, or `sim.engine` rewrite.

## Historical preregistration (retained verbatim below)

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

## Amendment 2026-09-29 — calib/eval split, recall metric, Δ win rule

No policy results exist; B0/B2 implementations live with Helen. This locks
how they (and any candidate) will be scored, before any of them run.

- Split: 15 calibration prompts (3/domain) for ALL fitting and tuning;
  20 heldout prompts (4/domain) are the eval set, every domain in both.
  B0 and B2 are built from calibration only. Eval prompts are never touched
  during design; first contact is scoring.
- Main metric: prefetch recall@k — fraction of the 8 routed experts present
  in the first k predictions — at budgets k ∈ {8, 12}, pooled over eval
  positions per trace. k=8 is exact-size prediction; k=12 allows 4 extra
  loads. Rank-blind by construction: only set membership counts.
- Win rule for recall gain Δ (candidate minus baseline), enforced by
  `sim/metrics.py::beats_baseline` (unit-tested): (1) Δ positive on BOTH
  traces; (2) min(Δ_ref, Δ_FLA) > |Δ_ref − Δ_FLA| (the smaller win must
  exceed the cross-trace wobble of Δ itself); (3) 95% prompt-level
  bootstrap CIs on Δ (resample the 20 eval prompts with replacement,
  B=10,000, seed 0, pooled positions per replicate) exclude zero on BOTH.
- Secondary: exact-set predictor accuracy (amendment-1 F rule: win on both
  traces by > F) and per-domain recall/accuracy breakdowns.
- The 87.16% cross-trace agreement is reported as the MEASUREMENT CEILING
  (no cross-trace comparison can resolve what the traces don't share),
  never as a baseline to beat.

## Amendment 2026-09-29 — headline comparison, Holm secondaries, frozen split

Confirmed: Δ-wobble only, no stacking. Both policies run on the same
trace, so shared trace-level noise cancels in the paired Δ (the paired
t-test argument); per-policy max(W) would count noise that cannot change
the winner. Deliberate, locked.

- Headline (the paper/Lossfunk claim): early-router vs B2
  (co-occurrence), recall@8. Single pre-specified primary; rules (1)(2)(3)
  from amendment 2, no multiplicity adjustment.
- Secondaries (5 tests: vs B0@8, vs B1@8, vs B0@12, vs B1@12, vs B2@12):
  rules (1)(2) per test, and rule (3) replaced by Holm step-down on
  one-sided prompt-level bootstrap p-values at family α=0.05
  (`sim/metrics.py::holm_reject`). Picked Holm over exploratory-only so
  secondary claims stay possible with FWER control. B0/B1/B2 are Helen's;
  B2 = co-occurrence.
- Frozen split (no seeds-pick-them, IDs fixed here). Calibration (15):
  chat_calib_01..03, code_calib_01..03, math_calib_01..03,
  sum_calib_01..03, tamil_calib_01..03. Eval (20): chat_held_01..04,
  code_held_01..04, math_held_01..04, sum_held_01..04, tamil_held_01..04.
  Every domain in both halves.
- Per-domain numbers (n≈4 eval prompts each) are DESCRIPTIVE ONLY: report
  them, make no significance claims on them. Locked so nobody is tempted
  later.
- FREEZE: after this amendment, any rule change needs a written reason AND
  must precede the affected policy runs. The preregistration is finished.

## Amendment 2026-09-29 — fused-event replay, slot-metric correction
## (the "Amendment 3" item from review; filed as 4 since 3 exists)

No policy runs exist. Written reason: review-identified gaps, pre-policy.

- Fused-event replay (matters most): each (prompt, token, layer) position
  replays as ONE event demanding all 8 routed experts simultaneously
  (atomic set demand against the resident set: hit iff routed ⊆ resident).
  NEVER as 8 sequential per-expert accesses — sequential replay lets
  recency policies score phantom hits across sub-accesses and flips
  LRU-vs-LFU rankings (August 2026 trace-evaluation paper). Enforced by
  `sim/engine.py::ReplayEvent` (frozen, exactly-8) and
  `events_from_routed` (one event per position; fragments rejected, never
  split or padded), unit-tested.
- Slot-metric correction: the old `slot_drift` compared RANKED lists, so it
  was rank-based despite the sets-not-ranks lock. Renamed
  `rank_slot_drift` (diagnostic only, never cited unqualified). New
  `set_expert_drift` = mean |A∖B|/8 over positions (set-based, per-expert
  units). Pilot parenthetical corrected: pilot set-expert drift is 1.14%
  (not the retired 7.01% rank figure); 35-prompt value is 1.71%.
- Tamil noise verdict (measurement, not a rule): perturbation magnitude on
  Tamil tokens is ~2x (mean|centered Δlogp| 1.75e-2 vs ~9e-3; median
  1.18e-2 vs ~5e-3), and logit-unit rebinning preserves Tamil's excess
  flips at fixed gap. B (bigger noise) is confirmed as at least part of
  the mechanism; tail-packing stays a possible secondary. The paragraph
  must say fixed-magnitude is NOT established.

## Amendment 2026-09-29 — D2 resolved: headline E4 bound (replaces Amendment 3 headline)

Written reason (freeze clause; no policy has run): (1) SpecMD §5.3, read
in full after Amendment 3, shows recall superiority does not transfer to
cache superiority — a recall headline cannot carry the paper's OS+bound
claim (D1-A). (2) The contribution is the bound; the headline must be the
bound. Supersedes the Amendment-3 headline only; everything else stands.
- Headline: E4. The queue's worst-case bound holds under an always-wrong
  predictor. Win criterion (deterministic, no multiplicity adjustment):
  max added delay to any real fetch ≤ the cap, on BOTH traces, all 20
  eval prompts. A bound holds or it doesn't.
- Recall comparisons demoted intact: early-router vs B2 @8 becomes first
  secondary; the other four keep Holm at family α=0.05. Rules (1)(2)
  unchanged everywhere.
- B0 = most-popular-8, B1 = previous-token's-8, B2 = co-occurrence
  tables (EdgeMoE method).
