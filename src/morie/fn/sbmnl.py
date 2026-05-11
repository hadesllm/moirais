# morie.fn — function file (hadesllm/morie)
"""Monopoly bargaining spatial.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbmnl(data=None, n=50):
    """Monopoly bargaining spatial.

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


short = "sbmnl"
alias = "sbmnl"
quote = "The spice must flow. -- Paul Atreides"
sbmnl = sbmnl


def cheatsheet() -> str:
    return "sbmnl({}) -> Monopoly bargaining spatial."
