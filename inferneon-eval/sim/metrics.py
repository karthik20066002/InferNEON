"""Locked scoring metrics (PREREGISTRATION.md amendments 1-2). Pure stdlib.

All comparisons are SET-based and rank-blind. Positions from different
prompts are never pooled inside a bootstrap replicate: replicates resample
whole prompts (prompt-level bootstrap), because positions within a prompt
are correlated.
"""
from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0


def recall_at_k(routed: set[int], predicted: Sequence[int], k: int) -> float:
    """Fraction of the routed experts present in the first k predictions."""
    if not routed:
        raise ValueError('empty routed set')
    if k < 1:
        raise ValueError('k must be positive')
    return len(set(predicted[:k]) & routed) / len(routed)


def exact_set_accuracy(routed: set[int], predicted: Sequence[int], k: int = 8) -> float:
    """1.0 iff the first-k predicted set equals the routed set exactly."""
    return 1.0 if set(predicted[:k]) == routed else 0.0


def mean_recall(per_prompt: Mapping[str, list[float]]) -> float:
    """Pooled mean over positions (prompts contribute proportionally)."""
    vals = [v for scores in per_prompt.values() for v in scores]
    if not vals:
        raise ValueError('no positions')
    return statistics.mean(vals)


def bootstrap_delta_ci(a: Mapping[str, list[float]], b: Mapping[str, list[float]],
                       replicates: int = BOOTSTRAP_REPLICATES,
                       seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% prompt-level bootstrap CI for mean(A) - mean(B), pooled positions.

    a/b map prompt_id -> per-position scores on the SAME trace and SAME
    positions. Replicates resample prompt ids with replacement; each
    replicate pools that replicate's prompts' positions. Deterministic in
    `seed`. Returns (2.5th, 97.5th) percentiles of the replicate deltas.
    """
    if set(a) != set(b):
        raise ValueError('policies must cover the same prompts')
    prompts = sorted(a)
    for p in prompts:
        if len(a[p]) != len(b[p]):
            raise ValueError(f'position mismatch in prompt {p}')
    rng = random.Random(seed)
    deltas = []
    for _ in range(replicates):
        sample = [rng.choice(prompts) for _ in prompts]
        sa = [v for p in sample for v in a[p]]
        sb = [v for p in sample for v in b[p]]
        deltas.append(statistics.mean(sa) - statistics.mean(sb))
    ordered = sorted(deltas)
    lo = ordered[int(0.025 * replicates)]
    hi = ordered[int(0.975 * replicates) - 1]
    return lo, hi


def wobble(metric_ref: float, metric_fla: float) -> float:
    """Absolute cross-trace difference of one policy's metric (amendment 1)."""
    return abs(metric_ref - metric_fla)


def bootstrap_delta_p(a: Mapping[str, list[float]], b: Mapping[str, list[float]],
                      replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED) -> float:
    """One-sided prompt-level bootstrap p-value for mean(A) > mean(B).

    Same resampling as bootstrap_delta_ci. p = fraction of replicate deltas
    <= 0, with +1 smoothing (never exactly 0). Deterministic in `seed`.
    Used for Holm-corrected secondary comparisons (amendment 3).
    """
    if set(a) != set(b):
        raise ValueError('policies must cover the same prompts')
    prompts = sorted(a)
    for p in prompts:
        if len(a[p]) != len(b[p]):
            raise ValueError(f'position mismatch in prompt {p}')
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        sample = [rng.choice(prompts) for _ in prompts]
        sa = [v for p in sample for v in a[p]]
        sb = [v for p in sample for v in b[p]]
        if statistics.mean(sa) - statistics.mean(sb) <= 0:
            extreme += 1
    return (1 + extreme) / (replicates + 1)


def holm_reject(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down FWER control at family level `alpha`.

    Sorts p-values ascending; rejects while p_(i) <= alpha/(m-i+1), stops at
    the first failure. Returns reject flags aligned with input order.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def beats_baseline(delta_ref: float, delta_fla: float,
                   ci_ref: tuple[float, float],
                   ci_fla: tuple[float, float]) -> dict:
    """Amendment-2 win rule for recall gain Δ (candidate minus baseline).

    (1) Δ positive on both traces; (2) the smaller Δ exceeds the cross-trace
    wobble |Δ_ref − Δ_FLA|; (3) both 95% prompt-level bootstrap CIs exclude
    zero. Returns per-criterion booleans plus the verdict.
    """
    c1 = delta_ref > 0 and delta_fla > 0
    c2 = min(delta_ref, delta_fla) > abs(delta_ref - delta_fla)
    c3 = ci_ref[0] > 0 and ci_fla[0] > 0
    return {'positive_on_both': c1, 'exceeds_wobble': c2,
            'ci_excludes_zero': c3, 'verdict': c1 and c2 and c3}
