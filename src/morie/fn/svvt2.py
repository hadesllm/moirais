"""2D vote trading equilibrium"""

import numpy as np

from ._containers import DescriptiveResult


def vote_trade_2d(x, *, ideal_point=None):
    """2D vote trading equilibrium

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
        name="svvt2",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


vote = vote_trade_2d


def cheatsheet() -> str:
    return "vote_trade_2d({}) -> 2D vote trading equilibrium"
