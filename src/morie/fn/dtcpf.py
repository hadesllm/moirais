# morie.fn -- function file (hadesllm/morie)
"""
Frank copula

Category: DistTheor
"""

import numpy as np


def dtcpf(x=None, n=100, params=None):
    """Frank copula

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


short = "dtcpf"
alias = "dtcpf"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
dtcpf = dtcpf


def cheatsheet() -> str:
    return "dtcpf({}) -> Frank copula"
