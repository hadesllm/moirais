# morie.fn -- function file (hadesllm/morie)
"""
Max-stable process

Category: DistTheor
"""

import numpy as np


def dtmxs(x=None, n=100, params=None):
    """Max-stable process

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if x is None:
        x = np.random.default_rng(0).standard_normal(n)
    if params is None:
        params = {"loc": float(np.mean(x)), "scale": float(np.std(x))}
    stat = float(np.mean(x))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x)), "params": params},
    )


short = "dtmxs"
alias = "dtmxs"
quote = "No man ever steps in the same river twice. -- Heraclitus"
dtmxs = dtmxs


def cheatsheet() -> str:
    return "dtmxs({}) -> Max-stable process"
