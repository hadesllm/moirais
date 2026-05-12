# morie.fn -- function file (hadesllm/morie)
"""Sorting polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plsor(data=None, n=50):
    """Sorting polarization index.

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


short = "plsor"
alias = "plsor"
quote = "The spice must flow. -- Paul Atreides"
plsor = plsor


def cheatsheet() -> str:
    return "plsor({}) -> Sorting polarization index."
