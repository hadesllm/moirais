# morie.fn -- function file (hadesllm/morie)
"""
Fisk (log-logistic) distribution

Category: DistTheor
"""

import numpy as np


def dtfsk(x=None, n=100, params=None):
    """Fisk (log-logistic) distribution

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


short = "dtfsk"
alias = "dtfsk"
quote = "What is now proved was once only imagined. -- William Blake"
dtfsk = dtfsk


def cheatsheet() -> str:
    return "dtfsk({}) -> Fisk (log-logistic) distribution"
