# morie.fn -- function file (hadesllm/morie)
"""Issue attention model.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def isatt(data=None, n=50):
    """Issue attention model.

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


short = "isatt"
alias = "isatt"
quote = "The spice must flow. -- Paul Atreides"
isatt = isatt


def cheatsheet() -> str:
    return "isatt({}) -> Issue attention model."
