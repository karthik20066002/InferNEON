# Prefetch-deadline guess (locked BEFORE running the test)

> RETIRED: this unequal-L arithmetic is not a single-lane result. The test
> now checks matched timing and the shared perfect-ID function instead.
> See the current scope in `PREREGISTRATION.md`; retain the text below only
> as a record of the rejected setup, not as an experiment that can speak.

Date: 2026-09-30. Test: `tests/test_prefetch_deadline.py`.

## Question

Does a perfect expert predictor that speaks too late lose to an
imperfect predictor that speaks early, once fetch bandwidth and
execution deadlines are counted?

This is the hinge between the old drift numbers (12.84% of spots
change sets, 1.71% of fetched experts change over 149,640 pooled
positions) and the memory-contract thesis: IDs alone may rank systems
wrong; IDs + size + deadline + residency may rank them right.

## Setup (all documented in the test)

- 4 fused 8-expert demands, consecutive overlap 4/8 (so a stale
  previous-demand predictor scores 0.5 recall after a cold start).
- A = oracle: predicts every next demand exactly (recall 1.0),
  lookahead 0 events (issued at demand).
- B = stale (B1 analogue: previous demand), mean recall 0.375,
  lookahead 4 events (one synthetic pass early; analogue of the
  40-event forward pass in `scripts/e1_lite.py`).
- Bandwidth `B = 400e6` bytes/s from `sim/engine.py::SimConfig`,
  `EXPERT_BYTES = 1572864` from `sim/eviction.py`.
  Per-expert fetch = EXPERT_BYTES / B ≈ 3.93 ms.
- Compute slack (ASSUMED, test-local): 8 ms per event.
  A hides 0 ms. B hides up to 32 ms for correctly predicted bytes.
- Empty cache, demand paging. Eviction held fixed (not the variable).
  Recall via `sim/metrics.py::recall_at_k` (set-based, rank-blind).
  Fused demands via `sim/engine.py::events_from_routed`.

## Original guess

A wins mean recall (1.0 > 0.375) but LOSES on total exposed I/O:
A exposes ~125.8 ms, B exposes ~78.6 ms (event 0 full miss + 3 x
half-miss). Ranking inverts: accuracy ranks A first, deadlines rank
B first.

## Falsification

If total unhidden A <= total unhidden B under the numbers above, the
guess is WRONG (arithmetic error or model bug — say so, don't rescue
it). If the test passes, it is still only a synthetic existence proof:
it shows accuracy != performance CAN happen, not that it DOES happen
on Qwen traces or CM4 hardware. That transfer is the fellowship work.

## Ties to old InferNEON work (not a new direction)

- Fused 8-set demands: firmware dispatch + `sim/engine.py::ReplayEvent`.
- B1 previous-token's-8: prereg baseline family.
- Identity + size hints: layer table / uniform-expert invariant,
  `EXPERT_BYTES`.
- Bandwidth/deadline knobs: `SimConfig.B/t_req/Q_dev`,
  `TokenResult.unhidden_io_ms` (stub pending — this test fills it).
- Backends: QEMU virtio now, eMMC/PCIe later; RTX vs CM4-class as the
  hierarchy experiment.
