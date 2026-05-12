# morie.fn -- function file (hadesllm/morie)
"""Delay cost bargaining.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbdel(data=None, n=50):
    """Delay cost bargaining.

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


short = "sbdel"
alias = "sbdel"
quote = "The spice must flow. -- Paul Atreides"
sbdel = sbdel


def cheatsheet() -> str:
    return "sbdel({}) -> Delay cost bargaining."
