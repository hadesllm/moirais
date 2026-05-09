# moirais.fn — function file (hadesllm/moirais)
"""
Hadley cell expansion

Category: GeoClim
"""

import numpy as np


def gchad(data=None, coords=None, n=50):
    """Hadley cell expansion

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


short = "gchad"
alias = "gchad"
quote = "Go beyond! Plus Ultra! -- All Might"
gchad = gchad


def cheatsheet() -> str:
    return "gchad({}) -> Hadley cell expansion"
