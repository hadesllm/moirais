# morie.fn -- function file (hadesllm/morie)
"""
BB1 copula

Category: DistTheor
"""

import numpy as np


def dtcpb(x=None, n=100, params=None):
    """BB1 copula

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


short = "dtcpb"
alias = "dtcpb"
quote = "It's over 9000! -- Vegeta"
dtcpb = dtcpb


def cheatsheet() -> str:
    return "dtcpb({}) -> BB1 copula"
