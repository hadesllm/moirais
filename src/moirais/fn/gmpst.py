# moirais.fn — function file (hadesllm/moirais)
"""Geometric mean probability statistic."""

from __future__ import annotations

from ._containers import DescriptiveResult


def gmp_statistic(predicted_probs, observed) -> DescriptiveResult:
    """Geometric mean probability of correct predictions.

    .. epigraph:: "Tread lightly." -- Walter White, Breaking Bad
    """
    import numpy as np

    p = np.asarray(predicted_probs, dtype=float)
    y = np.asarray(observed, dtype=float)
    correct_probs = np.where(y == 1, p, 1.0 - p)
    correct_probs = np.clip(correct_probs, 1e-15, 1.0)
    log_gmp = np.mean(np.log(correct_probs))
    gmp = float(np.exp(log_gmp))
    return DescriptiveResult(
        name="gmp_statistic",
        value=gmp,
        extra={"log_gmp": float(log_gmp), "n": len(p)},
    )


gmpst = gmp_statistic


def cheatsheet() -> str:
    return "gmp_statistic({}) -> Geometric mean probability statistic."
