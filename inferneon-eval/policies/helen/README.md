# policies/helen — Helen owns the real policies. AI ships interfaces + stubs + conformance tests only.

Modules (default per brief §2):
- `lfu_halving.py` — decayed LFU with halving every W accesses (W tuned on calibration)
- `admission.py` — FETCH vs ADMIT: keep (evict victim) or discard after use; predicted land staging-only
- `speculation_controller.py` — which predictions become fetches, release/cancel, rules R1-R5 independently togglable
- `budget_allocator.py` — per-layer slot allocation (greedy vs UCP-lookahead live in baselines; Helen's allocator here)
- `predictor_scheduler.py` — ranked (layer, expert, confidence) + availability time (input-ready + compute)

Each file contains: abstract interface with docstrings (inputs/outputs/when-called/must-not-touch),
trivial reference stub (always-admit / no-prediction / etc.), and a conformance suite under `tests/`.
Do NOT write real implementations here.
