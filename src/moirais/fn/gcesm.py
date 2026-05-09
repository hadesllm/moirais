# moirais.fn — function file (hadesllm/moirais)
"""
Earth system model spatial

Category: GeoClim
"""

import numpy as np


def gcesm(data=None, coords=None, n=50):
    """Earth system model spatial

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


short = "gcesm"
alias = "gcesm"
quote = "Make it so. -- Picard"
gcesm = gcesm


def cheatsheet() -> str:
    return "gcesm({}) -> Earth system model spatial"
