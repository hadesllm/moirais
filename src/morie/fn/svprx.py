"""Proximity voting model (Euclidean).

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def svprx(data=None, n=50):
    """Proximity voting model (Euclidean).

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


short = "svprx"
alias = "svprx"
quote = "The spice must flow. -- Paul Atreides"
svprx = svprx


def cheatsheet() -> str:
    return "svprx({}) -> Proximity voting model (Euclidean)."
