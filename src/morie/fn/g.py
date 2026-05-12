# morie.fn -- function file (hadesllm/morie)
"""Hedges' g effect size (bias-corrected Cohen's d)."""

import math
from typing import Union

import numpy as np

from morie.fn.d import cohens_d


def hedges_g(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
) -> float:
    """
    Hedges' g effect size (bias-corrected Cohen's d).

    Applies Hedges' correction factor J(m) to reduce small-sample bias:
    g = d * J(m),  where m = n1 + n2 - 2
    J(m) = 1 - 3 / (4m - 1)

    :param x1: First group sample.
    :param x2: Second group sample.
    :return: Hedges' g (signed float).
    :raises ValueError: If either sample has fewer than 2 observations.

    References
    ----------
    Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
        size and related estimators. Journal of Educational Statistics, 6(2), 107-128.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 2:
        raise ValueError("x1 must have at least 2 observations.")
    if len(a2) < 2:
        raise ValueError("x2 must have at least 2 observations.")
    d = cohens_d(a1, a2, pooled=True)
    m = len(a1) + len(a2) - 2
    # Hedges correction factor: exact via gamma functions; approximate for large m
    if m > 0:
        j = math.exp(math.lgamma(m / 2) - math.log(math.sqrt(m / 2)) - math.lgamma((m - 1) / 2))
    else:
        j = 1.0
    return float(d * j) if math.isfinite(d) else float("nan")


g = hedges_g


def cheatsheet() -> str:
    return "hedges_g({}) -> Hedges' g effect size (bias-corrected Cohen's d)."
