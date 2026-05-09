"""Spatial utility maximizer"""

import numpy as np

from ._containers import DescriptiveResult


def utility_max(x, *, ideal_point=None):
    """Spatial utility maximizer

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
        name="svutm",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


util = utility_max


def cheatsheet() -> str:
    return "utility_max({}) -> Spatial utility maximizer"
