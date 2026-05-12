# morie.fn -- function file (hadesllm/morie)
"""
Flow direction raster

Category: HydroSp
"""

import numpy as np


def hyfdr(flow=None, precip=None, coords=None, n=50):
    """Flow direction raster

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if flow is None:
        flow = np.abs(np.random.default_rng(0).standard_normal(n)) * 100
    if precip is None:
        precip = np.abs(np.random.default_rng(1).standard_normal(n)) * 50
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    stat = float(np.mean(flow))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(flow), "mean_flow": float(np.mean(flow)), "mean_precip": float(np.mean(precip))},
    )


short = "hyfdr"
alias = "hyfdr"
quote = "Believe it! -- Naruto"
hyfdr = hyfdr


def cheatsheet() -> str:
    return "hyfdr({}) -> Flow direction raster"
