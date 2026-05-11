# morie.fn — function file (hadesllm/morie)
"""Correct classification rate."""

from __future__ import annotations

from ._containers import DescriptiveResult


def fit_statistic_correct(predicted, observed) -> DescriptiveResult:
    """Proportion of correctly classified votes.

    .. epigraph:: "Respect the chemistry." -- Walter White, Breaking Bad
    """
    import numpy as np

    pred = np.asarray(predicted, dtype=float).round()
    obs = np.asarray(observed, dtype=float)
    correct = int(np.sum(pred == obs))
    n = len(obs)
    rate = float(correct / max(n, 1))
    return DescriptiveResult(
        name="fit_statistic_correct",
        value=rate,
        extra={"correct": correct, "n": n, "rate": rate},
    )


fitst = fit_statistic_correct


def cheatsheet() -> str:
    return "fit_statistic_correct({}) -> Correct classification rate."
