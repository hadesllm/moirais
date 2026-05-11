"""Convex optimization spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umcex(data=None, n=50):
    """Convex optimization spatial.

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


short = "umcex"
alias = "umcex"
quote = "The spice must flow. -- Paul Atreides"
umcex = umcex


def cheatsheet() -> str:
    return "umcex({}) -> Convex optimization spatial."
