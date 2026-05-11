"""City-block/L1 utility function.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svcut(data=None, n=50):
    """City-block/L1 utility function.

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


short = "svcut"
alias = "svcut"
quote = "The spice must flow. -- Paul Atreides"
svcut = svcut


def cheatsheet() -> str:
    return "svcut({}) -> City-block/L1 utility function."
