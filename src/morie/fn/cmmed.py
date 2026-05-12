# morie.fn -- function file (hadesllm/morie)
"""Committee median rule.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cmmed(data=None, n=50):
    """Committee median rule.

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


short = "cmmed"
alias = "cmmed"
quote = "The spice must flow. -- Paul Atreides"
cmmed = cmmed


def cheatsheet() -> str:
    return "cmmed({}) -> Committee median rule."
