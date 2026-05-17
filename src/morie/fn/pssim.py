# morie.fn -- function file (hadesllm/morie)
"""Simulation-based spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def pssim(data=None, n=50):
    """Simulation-based spatial.

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


short = "pssim"
alias = "pssim"
quote = "A journey of a thousand miles begins with a single step. -- Lao Tzu"
pssim = pssim


def cheatsheet() -> str:
    return "pssim({}) -> Simulation-based spatial."
