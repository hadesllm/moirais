# morie.fn — function file (hadesllm/morie)
"""
Generalized extreme value

Category: DistTheor
"""

import numpy as np


def dtgev(x=None, n=100, params=None):
    """Generalized extreme value

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


short = "dtgev"
alias = "dtgev"
quote = "People's dreams never end! -- Blackbeard"
dtgev = dtgev


def cheatsheet() -> str:
    return "dtgev({}) -> Generalized extreme value"
