# morie.fn — function file (hadesllm/morie)
"""Dynamic IRT model estimation"""

import numpy as np

from .._spatial_voting import dynamic_irt as _dynamic_irt
from ._containers import DescriptiveResult


def dynamic_irt_estimate(votes=None, *, time_periods=None, dims=1, n=10, m=5):
    """Dynamic IRT model for time-varying ideal points.

    Returns
    -------
    DescriptiveResult
    """
    if votes is None:
        rng = np.random.default_rng(0)
        votes = (rng.random((n, m)) > 0.5).astype(float)
    if time_periods is None:
        time_periods = np.arange(m)
    votes = np.asarray(votes, dtype=float)
    result = _dynamic_irt(votes, time_periods=time_periods)
    return DescriptiveResult(
        name="Dynamic IRT",
        value=result.get("loglik", 0.0),
        extra={"dims": dims, "n_periods": len(time_periods), **result},
    )


short = "dyirt"
alias = "dynamic_irt_estimate"
quote = "We are what we repeatedly do. Excellence is not an act, but a habit. — Aristotle"
dynamic_irt_estimate = dynamic_irt_estimate


def cheatsheet() -> str:
    return "dynamic_irt_estimate({}) -> Dynamic IRT model estimation"
