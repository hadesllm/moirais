# morie.fn — function file (hadesllm/morie)
"""Proximity-based coalition.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def cfprx(data=None, n=50):
    """Proximity-based coalition.

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


short = "cfprx"
alias = "cfprx"
quote = "The spice must flow. -- Paul Atreides"
cfprx = cfprx


def cheatsheet() -> str:
    return "cfprx({}) -> Proximity-based coalition."
