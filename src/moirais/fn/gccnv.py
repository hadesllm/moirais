# moirais.fn — function file (hadesllm/moirais)
"""
Convection change spatial

Category: GeoClim
"""

import numpy as np


def gccnv(data=None, coords=None, n=50):
    """Convection change spatial

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


short = "gccnv"
alias = "gccnv"
quote = "Bankai! -- Ichigo"
gccnv = gccnv


def cheatsheet() -> str:
    return "gccnv({}) -> Convection change spatial"
