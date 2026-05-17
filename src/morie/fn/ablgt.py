# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Light pollution spatial

Category: AirBio
"""

import numpy as np


def ablgt(data=None, coords=None, n=50):
    """Light pollution spatial

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


short = "ablgt"
alias = "ablgt"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
ablgt = ablgt


def cheatsheet() -> str:
    return "ablgt({}) -> Light pollution spatial"
