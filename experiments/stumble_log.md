# Stumble log — 30 September 2026, evening

> RETIRED as evidence: unequal lookahead and independent overlap arithmetic
> do not measure a serial storage lane. The historical observation below is
> not a result. Only `reports/exposed_wait_matched_L.json` supports the
> current exposed-wait comparison; the old executable test was replaced.

I was doing the obvious next thing: ranking two predictors by recall
over the drift traces. Oracle IDs versus previous-demand, B1-style.
Nothing ambitious. Just the next rung on the ladder.

As a sanity check I added fetch-time arithmetic — expert bytes and link
bandwidth already sitting in the repo — into the existing unhidden_io
stub. I expected it to agree with recall.

It did not.

## The flip

Same 4 fused 8-expert demands. Empty cache. Eviction held fixed.

- Perfect predictor, recall 1.0, issued at demand time: ~125.8 ms exposed.
- Stale predictor, recall ~0.375, issued a pass early: ~78.6 ms exposed.

The better guesser lost. The earlier guesser won.

Good sanity check. Not the answer I wanted.

## So the question changed

Route IDs aren't the only variable that sets system cost. Size,
deadline, residency matter — and they can outvote correctness.

New formulation: what is the smallest set of scheduling hints
(identity, size, reuse likelihood, execution deadline) a sparse model
must expose to make storage a practical execution tier?

## Receipts (all in this commit)

- experiments/prefetch_deadline_guess.md — the guess, written before running
- tests/test_prefetch_deadline.py — the test, synthetic only
- sim/engine.py — fused ReplayEvent contract, TokenResult.unhidden stub
- sim/eviction.py — EXPERT_BYTES, policies (held fixed here)
- README pipeline — dispatch set + barrier, step 5 resolve

The 1.7% drift result is demoted to the observation that led here:
route stability and fetch-relevance aren't the same variable.

Not a Qwen claim. That transfer is the work.
