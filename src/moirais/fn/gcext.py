# moirais.fn — function file (hadesllm/moirais)
"""
Extreme event frequency

Category: GeoClim
"""

import numpy as np


def gcext(data=None, coords=None, n=50):
    """Extreme event frequency

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


short = "gcext"
alias = "gcext"
quote = "Go beyond! Plus Ultra! -- All Might"
gcext = gcext


def cheatsheet() -> str:
    return "gcext({}) -> Extreme event frequency"
