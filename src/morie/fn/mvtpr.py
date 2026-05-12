# morie.fn -- function file (hadesllm/morie)
"""Proportional representation median.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtpr(data=None, n=50):
    """Proportional representation median.

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


short = "mvtpr"
alias = "mvtpr"
quote = "The spice must flow. -- Paul Atreides"
mvtpr = mvtpr


def cheatsheet() -> str:
    return "mvtpr({}) -> Proportional representation median."
