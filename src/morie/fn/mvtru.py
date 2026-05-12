# morie.fn -- function file (hadesllm/morie)
"""Runoff median voter.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtru(data=None, n=50):
    """Runoff median voter.

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


short = "mvtru"
alias = "mvtru"
quote = "The spice must flow. -- Paul Atreides"
mvtru = mvtru


def cheatsheet() -> str:
    return "mvtru({}) -> Runoff median voter."
