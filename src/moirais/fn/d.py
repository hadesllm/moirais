# moirais.fn — function file (hadesllm/moirais)
"""Cohen's d effect size for two independent groups."""

import math
from typing import Union

import numpy as np


def cohens_d(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    pooled: bool = True,
) -> float:
    """
    Cohen's d effect size for two independent groups.

    Using pooled SD (default):
    d = (mean_1 - mean_2) / s_pooled

    where s_pooled = sqrt([(n1-1)*s1^2 + (n2-1)*s2^2] / (n1+n2-2)).

    Using the control-group SD (``pooled=False``):
    d = (mean_1 - mean_2) / s_2

    Conventional benchmarks: |d| = 0.2 small, 0.5 medium, 0.8 large
    (Cohen, 1988).

    :param x1: First group sample.
    :param x2: Second group sample (reference/control when pooled=False).
    :param pooled: If True use pooled SD denominator. Default True.
    :return: Cohen's d (signed float).
    :raises ValueError: If either sample has fewer than 2 observations.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
        (2nd ed.). Lawrence Erlbaum Associates.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 2:
        raise ValueError("x1 must have at least 2 observations.")
    if len(a2) < 2:
        raise ValueError("x2 must have at least 2 observations.")
    mean_diff = np.mean(a1) - np.mean(a2)
    if pooled:
        n1, n2 = len(a1), len(a2)
        s1_sq = np.var(a1, ddof=1)
        s2_sq = np.var(a2, ddof=1)
        s_pooled = math.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
        return float(mean_diff / s_pooled) if s_pooled > 0 else float("nan")
    else:
        s2 = float(np.std(a2, ddof=1))
        return float(mean_diff / s2) if s2 > 0 else float("nan")


d = cohens_d


def cheatsheet() -> str:
    return "cohens_d({}) -> Cohen's d effect size for two independent groups."
