"""Median voter theorem test"""

import numpy as np

from ._containers import DescriptiveResult


def median_voter(x, *, ideal_point=None):
    """Median voter theorem test

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    ideal = np.asarray(ideal_point, dtype=float) if ideal_point is not None else np.zeros_like(x)
    diff = x - ideal
    dist_sq = float(np.sum(diff**2))
    val = np.exp(-0.5 * dist_sq)
    return DescriptiveResult(
        name="svmvt",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


medi = median_voter


def cheatsheet() -> str:
    return "median_voter({}) -> Median voter theorem test"
