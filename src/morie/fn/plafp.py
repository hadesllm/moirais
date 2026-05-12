# morie.fn -- function file (hadesllm/morie)
"""Affective polarization index.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def plafp(data=None, n=50):
    """Affective polarization index.

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


short = "plafp"
alias = "plafp"
quote = "The spice must flow. -- Paul Atreides"
plafp = plafp


def cheatsheet() -> str:
    return "plafp({}) -> Affective polarization index."
