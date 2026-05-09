# moirais.fn — function file (hadesllm/moirais)
"""
UV-B radiation spatial

Category: EnvStat
"""

import numpy as np


def enuvb(data=None, coords=None, n=50):
    """UV-B radiation spatial

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


short = "enuvb"
alias = "enuvb"
quote = "I alone level up. -- Sung Jin-Woo"
enuvb = enuvb


def cheatsheet() -> str:
    return "enuvb({}) -> UV-B radiation spatial"
