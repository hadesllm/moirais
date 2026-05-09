"""Dot-product directional vote.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svdot(data=None, n=50):
    """Dot-product directional vote.

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


short = "svdot"
alias = "svdot"
quote = "The spice must flow. -- Paul Atreides"
svdot = svdot


def cheatsheet() -> str:
    return "svdot({}) -> Dot-product directional vote."
