"""Normal kernel vote probability"""

import numpy as np

from ._containers import DescriptiveResult


def normal_vote(x, *, ideal_point=None):
    """Normal kernel vote probability

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
        name="svnrm",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


norm = normal_vote


def cheatsheet() -> str:
    return "normal_vote({}) -> Normal kernel vote probability"
