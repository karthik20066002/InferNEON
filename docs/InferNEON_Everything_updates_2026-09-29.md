# InferNEON_Everything.pdf — revision notes (2026-09-29, second pass)

Source: cross-check of the 29-Sep PDF against arXiv/HF/official docs (read
in full where load-bearing) and repo evidence. Apply to the source doc,
re-export, re-pin (the `49a6c34` pin already misses the memo, capture
scripts, timing evidence, and refusal fix — re-pin to current HEADs once
the capture lands).

## Decisions closed since the PDF (write in as decided)

- **D3 → 128 tokens.** Cold-disk timing done (drop_caches,
  page_cold=True): decode flat ~3.6s/token all modes; prefill 56–61s
  (warm 26s). 128 ≈ 8 min/prompt ≈ 5h over 35 prompts; 512 ≈ 19h.
  Evidence: `reports/time_decode_{off,cold,warm}.json`.
- **D4 → A (Helen writes all five).** The "Helen-owned stubs" split is
  dropped; implementation is testing-led throughout.
- **D1 → A confirmed viable.** Predictor prior work verified to the
  mechanism level (Eliseev §3.2: next-layer gate on previous-layer state).
  Cite, don't claim.
- **sudo resolved** via scoped NOPASSWD rule (drop_caches write only).
- **Filler priority (new, firmware scheduling):** unpack arrived tiles >
  next-layer router prediction > KV housekeeping. Open sub-question:
  reserve predictor slices under sustained misses, or let E4 measure the
  death spiral first (recommended: measure first).

## D2 → A (headline E4 bound), Amendment 5 in PREREGISTRATION.md

Written reason (freeze clause, no policy has run): SpecMD §5.3, read in
full after Amendment 3, shows recall superiority does not transfer to
cache superiority — a recall headline cannot carry the OS+bound claim.
Headline: E4 adheres — max added delay to any real fetch ≤ cap, both
traces, all 20 eval prompts. Early-router vs B2 @8 becomes first
secondary; other four recall comparisons keep Holm at family α=0.05.

## Prior-art table corrections (§4)

- **SpecMD row: soften.** Full text shows a prefetch queue + watchdog
  gated on capacity. True differentiators: no priority tiers, no
  in-flight cap, no cancellation, no *provable* bound (theirs is empirical
  TTFT). E4 (always-wrong-predictor bound test) has no counterpart there.
- **Add (E1-shaping):** LRU/LFU "perform quite poorly"; Least-Stale up to
  85× better on collision misses. E1 must *expect* weak classical
  baselines; LFU-halving and Helen-dual are untested anywhere.
- **Add (E3-shaping):** §5.3 recall-vs-hit-rate dissociation.
- **EdgeMoE: add second half** (its own ahead-of-time expert prediction;
  harmless to D1, in D7 scope).
- **EdgeXpert 28nm: confirmed** (Samsung 28nm @800MHz, MICRO 2026) — a
  synthesized token-speculation + EMA co-design, not an OS.
- **Trace paper: confirmed** (Yu Zhang, v1 Aug 8 2026; v4 metadata-only).
- Pre-gated/Flash/TinyLFU/UCP/lookahead links: not individually checked.

## Findings precision (§6.2–6.3)

- "Almost always the weakest 8th pick": strict-8th-only is **38.9%** of
  set-changing spots (7,478/19,221, all 35 prompts). Near-universal is
  single-expert swaps (mean 1.07 replaced/spot) — reword to that.
- Permutation p<0.001 + "at equal length the gap remains": **no such
  analysis in the repo** — attach the method or drop to descriptive.
- §6.3 mechanism: B confirmed (Tamil perturbations ~2× in logit units,
  `scripts/noise_magnitude.py`); tail-packing secondary at best.
  "Fixed-magnitude" must not appear.
- Tamil token cost verified: 161.3 vs 93.0 position-weighted. 40,960
  requests ✓. 1-in-60 ✓ (1/0.0171 = 58.5).

## Hardware (D6 + §10.1)

- Confirmed: CM4 PCIe Gen 2 x1, Lite = no eMMC, ~413 MB/s plausible.
- **Open:** M.2 length support varies by carrier — need the exact
  Waveshare board model (D6).
- **Conflict:** the firmware sketch fetches from **eMMC** while the plan
  assumes Lite + NVMe. Which CM4 variant? (eMMC peaks ~100 MB/s.)

## Least-Stale vs the hybrid (E1 framing)

- LS (SpecMD §4): stale/current pass partition; victims from stale,
  current-pass unevictable except on OOM. Structural fix for collision
  misses; frequency-blind within queues (FIFO).
- Helen-dual weakness vs LS: MFU can shelter stale-hot experts while the
  warm tier churns current-pass ones.
- Hybrid (`sim/eviction.py::HybridStaleHeat`, Helen's permission):
  LS stale-pool bottom + heat promotion, halves split kept. E1-lite
  running now gives the preliminary verdict on drift data; capture
  traces give the decode-regime rematch.

## Paper-trail rows to add

- RAM memo + GPU cache in capture (~7×, hashes unchanged `de6811`).
- Refusal-scope fix (code-only, reports excluded).
- Cold-disk timing + D3→128 + D4→A + sudo resolution.
- Filler priority order + E4-spiral question.
- Capture launch 35×128 (in progress at time of writing).
- This cross-check (prior-art verdicts above).
- Amendment 5 (D2→A).
