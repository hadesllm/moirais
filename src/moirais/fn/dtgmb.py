# moirais.fn — function file (hadesllm/moirais)
"""
Gumbel distribution

Category: DistTheor
"""

import numpy as np


def dtgmb(x=None, n=100, params=None):
    """Gumbel distribution

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


short = "dtgmb"
alias = "dtgmb"
quote = "You should enjoy the detours. -- Ging"
dtgmb = dtgmb


def cheatsheet() -> str:
    return "dtgmb({}) -> Gumbel distribution"
