# morie.fn -- function file (hadesllm/morie)
"""Variance-based polarization.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plvar(data=None, n=50):
    """Variance-based polarization.

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


short = "plvar"
alias = "plvar"
quote = "The spice must flow. -- Paul Atreides"
plvar = plvar


def cheatsheet() -> str:
    return "plvar({}) -> Variance-based polarization."
