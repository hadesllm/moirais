# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""
Endotoxin spatial

Category: AirBio
"""

import numpy as np


def abend(data=None, coords=None, n=50):
    """Endotoxin spatial

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


short = "abend"
alias = "abend"
quote = "Arise. -- Shadow Monarch"
abend = abend


def cheatsheet() -> str:
    return "abend({}) -> Endotoxin spatial"
