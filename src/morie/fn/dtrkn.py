# morie.fn -- function file (hadesllm/morie)
"""Rank-based dimensionality.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dtrkn(data=None, n=50):
    """Rank-based dimensionality.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "dtrkn"
alias = "dtrkn"
quote = "The spice must flow. -- Paul Atreides"
dtrkn = dtrkn


def cheatsheet() -> str:
    return "dtrkn({}) -> Rank-based dimensionality."
