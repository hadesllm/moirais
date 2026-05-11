"""Probit spatial voting probability"""

import numpy as np

from ._containers import DescriptiveResult


def probit_vote(x, *, ideal_point=None):
    """Probit spatial voting probability

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
        name="svprv",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


prob = probit_vote


def cheatsheet() -> str:
    return "probit_vote({}) -> Probit spatial voting probability"
