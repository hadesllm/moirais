"""Mixed-norm spatial utility"""

import numpy as np

from ._containers import DescriptiveResult


def mixed_utility(x, *, ideal_point=None):
    """Mixed-norm spatial utility

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
        name="svmxu",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


mixe = mixed_utility


def cheatsheet() -> str:
    return "mixed_utility({}) -> Mixed-norm spatial utility"
