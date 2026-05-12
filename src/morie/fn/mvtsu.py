# morie.fn -- function file (hadesllm/morie)
"""Supermajority median voter.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtsu(data=None, n=50):
    """Supermajority median voter.

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


short = "mvtsu"
alias = "mvtsu"
quote = "The spice must flow. -- Paul Atreides"
mvtsu = mvtsu


def cheatsheet() -> str:
    return "mvtsu({}) -> Supermajority median voter."
