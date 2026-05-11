# morie.fn — function file (hadesllm/morie)
"""Median voter theorem 1D.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvt1d(data=None, n=50):
    """Median voter theorem 1D.

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


short = "mvt1d"
alias = "mvt1d"
quote = "The spice must flow. -- Paul Atreides"
mvt1d = mvt1d


def cheatsheet() -> str:
    return "mvt1d({}) -> Median voter theorem 1D."
