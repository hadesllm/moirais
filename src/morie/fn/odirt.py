# morie.fn — function file (hadesllm/morie)
"""Ordinal IRT model estimation"""

import numpy as np

from .._spatial_voting import ordinal_irt as _ordinal_irt
from ._containers import DescriptiveResult


def ordinal_irt_estimate(responses=None, *, dims=1, n=10, m=5):
    """Ordinal IRT model for ordered categorical responses.

    Returns
    -------
    DescriptiveResult
    """
    if responses is None:
        rng = np.random.default_rng(0)
        responses = rng.integers(1, 6, size=(n, m)).astype(float)
    responses = np.asarray(responses, dtype=float)
    result = _ordinal_irt(responses, n_dims=dims)
    return DescriptiveResult(
        name="Ordinal IRT",
        value=result.get("loglik", 0.0),
        extra={"dims": dims, **result},
    )


short = "odirt"
alias = "ordinal_irt_estimate"
quote = "We are what we repeatedly do. Excellence is not an act, but a habit. — Aristotle"
ordinal_irt_estimate = ordinal_irt_estimate


def cheatsheet() -> str:
    return "ordinal_irt_estimate({}) -> Ordinal IRT model estimation"
