"""Block vote trading.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def vtblk(data=None, n=50):
    """Block vote trading.

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


short = "vtblk"
alias = "vtblk"
quote = "The spice must flow. -- Paul Atreides"
vtblk = vtblk


def cheatsheet() -> str:
    return "vtblk({}) -> Block vote trading."
