"""Median voter in 2D (Plott conditions)"""

import numpy as np

from ._containers import DescriptiveResult


def median_voter_2d(x, *, ideal_point=None):
    """Median voter in 2D (Plott conditions)

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
        name="svmv2",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


medi = median_voter_2d


def cheatsheet() -> str:
    return "median_voter_2d({}) -> Median voter in 2D (Plott conditions)"
