# morie.fn -- function file (hadesllm/morie)
"""Electoral importance weights.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def elimp(data=None, n=50):
    """Electoral importance weights.

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


short = "elimp"
alias = "elimp"
quote = "The spice must flow. -- Paul Atreides"
elimp = elimp


def cheatsheet() -> str:
    return "elimp({}) -> Electoral importance weights."
