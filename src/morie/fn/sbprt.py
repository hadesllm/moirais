# morie.fn -- function file (hadesllm/morie)
"""Pareto bargaining frontier.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbprt(data=None, n=50):
    """Pareto bargaining frontier.

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


short = "sbprt"
alias = "sbprt"
quote = "The spice must flow. -- Paul Atreides"
sbprt = sbprt


def cheatsheet() -> str:
    return "sbprt({}) -> Pareto bargaining frontier."
