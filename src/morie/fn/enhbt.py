# morie.fn -- function file (hadesllm/morie)
"""
Habitat suitability spatial

Category: EnvStat
"""

import numpy as np


def enhbt(data=None, coords=None, n=50):
    """Habitat suitability spatial

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


short = "enhbt"
alias = "enhbt"
quote = "The spice must flow. -- Paul Atreides"
enhbt = enhbt


def cheatsheet() -> str:
    return "enhbt({}) -> Habitat suitability spatial"
