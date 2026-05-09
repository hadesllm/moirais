"""Incentive-compatible utility max.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def uminc(data=None, n=50):
    """Incentive-compatible utility max.

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


short = "uminc"
alias = "uminc"
quote = "The spice must flow. -- Paul Atreides"
uminc = uminc


def cheatsheet() -> str:
    return "uminc({}) -> Incentive-compatible utility max."
