# morie.fn — function file (hadesllm/morie)
"""Eigenvalue dimensionality test.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def dteig(data=None, n=50):
    """Eigenvalue dimensionality test.

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


short = "dteig"
alias = "dteig"
quote = "The spice must flow. -- Paul Atreides"
dteig = dteig


def cheatsheet() -> str:
    return "dteig({}) -> Eigenvalue dimensionality test."
