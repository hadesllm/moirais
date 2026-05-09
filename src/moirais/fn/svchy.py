"""Cauchy kernel spatial voting"""

import numpy as np

from ._containers import DescriptiveResult


def cauchy_vote(x, *, ideal_point=None):
    """Cauchy kernel spatial voting

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
        name="svchy",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


cauc = cauchy_vote


def cheatsheet() -> str:
    return "cauchy_vote({}) -> Cauchy kernel spatial voting"
