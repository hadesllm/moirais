# morie.fn -- function file (hadesllm/morie)
"""Status quo point bargaining.

Category: Spatial
"""

import numpy as np

from ._containers import DescriptiveResult


def sbsqp(data=None, n=50):
    """Status quo point bargaining.

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


short = "sbsqp"
alias = "sbsqp"
quote = "The spice must flow. -- Paul Atreides"
sbsqp = sbsqp


def cheatsheet() -> str:
    return "sbsqp({}) -> Status quo point bargaining."
