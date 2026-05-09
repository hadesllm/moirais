"""Quadratic spatial utility function"""

import numpy as np

from ._containers import DescriptiveResult


def quad_utility(x, *, ideal_point=None):
    """Quadratic spatial utility function

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
        name="svqud",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


quad = quad_utility


def cheatsheet() -> str:
    return "quad_utility({}) -> Quadratic spatial utility function"
