"""Gaussian spatial utility function"""

import numpy as np

from ._containers import DescriptiveResult


def gauss_utility(x, *, ideal_point=None):
    """Gaussian spatial utility function

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
        name="svgau",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


gaus = gauss_utility


def cheatsheet() -> str:
    return "gauss_utility({}) -> Gaussian spatial utility function"
