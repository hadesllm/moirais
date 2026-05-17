"""Scalar-distance combined model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svsco(data=None, n=50):
    """Scalar-distance combined model.

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


short = "svsco"
alias = "svsco"
quote = "An investment in knowledge pays the best interest. -- Benjamin Franklin"
svsco = svsco


def cheatsheet() -> str:
    return "svsco({}) -> Scalar-distance combined model."
