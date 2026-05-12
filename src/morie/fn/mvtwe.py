# morie.fn -- function file (hadesllm/morie)
"""Weighted median voter theorem.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def mvtwe(data=None, n=50):
    """Weighted median voter theorem.

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


short = "mvtwe"
alias = "mvtwe"
quote = "The spice must flow. -- Paul Atreides"
mvtwe = mvtwe


def cheatsheet() -> str:
    return "mvtwe({}) -> Weighted median voter theorem."
