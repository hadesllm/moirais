"""Weighted voting game value"""

import numpy as np

from ._containers import DescriptiveResult


def weighted_vote(x, *, ideal_point=None):
    """Weighted voting game value

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
        name="svwvt",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


weig = weighted_vote


def cheatsheet() -> str:
    return "weighted_vote({}) -> Weighted voting game value"
