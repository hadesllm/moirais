"""Vote trading (logrolling) model"""

import numpy as np

from ._containers import DescriptiveResult


def vote_trading(x, *, ideal_point=None):
    """Vote trading (logrolling) model

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
        name="svvtr",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


vote = vote_trading


def cheatsheet() -> str:
    return "vote_trading({}) -> Vote trading (logrolling) model"
