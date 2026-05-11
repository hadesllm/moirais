# morie.fn — function file (hadesllm/morie)
"""Weighted Gini polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plwgn(data=None, n=50):
    """Weighted Gini polarization.

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


short = "plwgn"
alias = "plwgn"
quote = "The spice must flow. -- Paul Atreides"
plwgn = plwgn


def cheatsheet() -> str:
    return "plwgn({}) -> Weighted Gini polarization."
