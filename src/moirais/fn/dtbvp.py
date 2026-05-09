# moirais.fn — function file (hadesllm/moirais)
"""
Bivariate Poisson

Category: DistTheor
"""

import numpy as np


def dtbvp(x=None, n=100, params=None):
    """Bivariate Poisson

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


short = "dtbvp"
alias = "dtbvp"
quote = "Scatter, Senbonzakura. -- Byakuya"
dtbvp = dtbvp


def cheatsheet() -> str:
    return "dtbvp({}) -> Bivariate Poisson"
