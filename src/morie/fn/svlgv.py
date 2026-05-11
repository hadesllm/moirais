"""Logit spatial voting probability"""

import numpy as np

from ._containers import DescriptiveResult


def logit_vote(x, *, ideal_point=None):
    """Logit spatial voting probability

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
        name="svlgv",
        value=float(val),
        extra={"dist_sq": dist_sq},
    )


logi = logit_vote


def cheatsheet() -> str:
    return "logit_vote({}) -> Logit spatial voting probability"
