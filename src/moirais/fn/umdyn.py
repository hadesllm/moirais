"""Dynamic utility maximization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def umdyn(data=None, n=50):
    """Dynamic utility maximization.

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


short = "umdyn"
alias = "umdyn"
quote = "The spice must flow. -- Paul Atreides"
umdyn = umdyn


def cheatsheet() -> str:
    return "umdyn({}) -> Dynamic utility maximization."
