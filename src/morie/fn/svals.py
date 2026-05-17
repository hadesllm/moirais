"""Algebraic-distance (dot product) directional.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svals(data=None, n=50):
    """Algebraic-distance (dot product) directional.

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


short = "svals"
alias = "svals"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
svals = svals


def cheatsheet() -> str:
    return "svals({}) -> Algebraic-distance (dot product) directional."
