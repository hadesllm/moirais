"""Quadratic utility function for spatial voting.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svqut(data=None, n=50):
    """Quadratic utility function for spatial voting.

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


short = "svqut"
alias = "svqut"
quote = "The spice must flow. -- Paul Atreides"
svqut = svqut


def cheatsheet() -> str:
    return "svqut({}) -> Quadratic utility function for spatial voting."
