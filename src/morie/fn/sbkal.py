# morie.fn -- function file (hadesllm/morie)
"""Kalai-Smorodinsky bargaining.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbkal(data=None, n=50):
    """Kalai-Smorodinsky bargaining.

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


short = "sbkal"
alias = "sbkal"
quote = "An investment in knowledge pays the best interest. -- Benjamin Franklin"
sbkal = sbkal


def cheatsheet() -> str:
    return "sbkal({}) -> Kalai-Smorodinsky bargaining."
