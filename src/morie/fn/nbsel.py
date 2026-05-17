# morie.fn -- function file (hadesllm/morie)
"""
SEL sound exposure

Category: NoisBrd
"""

import numpy as np


def nbsel(data=None, coords=None, n=50):
    """SEL sound exposure

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(30, 90, n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean_db": float(np.mean(data)), "max_db": float(np.max(data))},
    )


short = "nbsel"
alias = "nbsel"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
nbsel = nbsel


def cheatsheet() -> str:
    return "nbsel({}) -> SEL sound exposure"
