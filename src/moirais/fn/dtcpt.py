# moirais.fn — function file (hadesllm/moirais)
"""
Student-t copula

Category: DistTheor
"""

import numpy as np


def dtcpt(x=None, n=100, params=None):
    """Student-t copula

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


short = "dtcpt"
alias = "dtcpt"
quote = "Fear is the mind-killer. -- Bene Gesserit"
dtcpt = dtcpt


def cheatsheet() -> str:
    return "dtcpt({}) -> Student-t copula"
