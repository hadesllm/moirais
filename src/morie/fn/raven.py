# morie.fn -- function file (hadesllm/morie)
"""Raven's progressive matrices scoring. 'Azarath Metrion Zinthos!' -- Raven"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def raven_score(
    responses: np.ndarray | list[int],
    correct: np.ndarray | list[int],
    *,
    n_sets: int = 5,
) -> DescriptiveResult:
    """Score a Raven's Progressive Matrices-style test.

    Computes total score, per-set scores, percentile estimate, and
    discrepancy analysis (comparing observed vs expected set performance
    given the total score).

    Parameters
    ----------
    responses : array
        Examinee responses (1-indexed answer choices).
    correct : array
        Correct answer key (same length as responses).
    n_sets : int
        Number of item sets (e.g. A-E for SPM).

    Returns
    -------
    DescriptiveResult
        ``value`` = total raw score.
    """
    responses = np.asarray(responses, dtype=int).ravel()
    correct = np.asarray(correct, dtype=int).ravel()
    if len(responses) != len(correct):
        raise ValueError("responses and correct must have the same length")
    n_items = len(responses)
    if n_items < n_sets:
        raise ValueError(f"Need at least {n_sets} items")
    scored = (responses == correct).astype(int)
    total = int(scored.sum())
    items_per_set = n_items // n_sets
    set_scores = []
    for s in range(n_sets):
        start = s * items_per_set
        end = start + items_per_set if s < n_sets - 1 else n_items
        set_scores.append(int(scored[start:end].sum()))
    expected_pct = total / n_items
    set_expected = [
        round(expected_pct * (items_per_set if s < n_sets - 1 else n_items - (n_sets - 1) * items_per_set), 1)
        for s in range(n_sets)
    ]
    discrepancy = [set_scores[s] - set_expected[s] for s in range(n_sets)]
    max_discrep = max(abs(d) for d in discrepancy)
    consistent = max_discrep <= 2
    return DescriptiveResult(
        name="Raven's Progressive Matrices score",
        value=total,
        extra={
            "n_items": n_items,
            "pct_correct": round(total / n_items * 100, 1),
            "set_scores": set_scores,
            "set_expected": set_expected,
            "discrepancy": [round(d, 1) for d in discrepancy],
            "consistent": consistent,
            "max_discrepancy": round(max_discrep, 1),
        },
    )


raven = raven_score


def cheatsheet() -> str:
    return "raven_score({}) -> Raven's progressive matrices scoring. 'Azarath Metrion Zinth"
