# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Dioxin spatial mapping

Category: AirBio
"""

import numpy as np


def abdxn(data=None, coords=None, n=50):
    """Dioxin spatial mapping

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "abdxn"
alias = "abdxn"
quote = "The heart has its reasons of which reason knows nothing. -- Blaise Pascal"
abdxn = abdxn


def cheatsheet() -> str:
    return "abdxn({}) -> Dioxin spatial mapping"
