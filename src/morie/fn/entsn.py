# morie.fn -- function file (hadesllm/morie)
"""
Tsunami risk spatial

Category: EnvStat
"""

import numpy as np


def entsn(data=None, coords=None, n=50):
    """Tsunami risk spatial

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


short = "entsn"
alias = "entsn"
quote = "The measure of a man is what he does with power. -- Plato"
entsn = entsn


def cheatsheet() -> str:
    return "entsn({}) -> Tsunami risk spatial"
