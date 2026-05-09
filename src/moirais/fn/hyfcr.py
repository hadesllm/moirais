# moirais.fn — function file (hadesllm/moirais)
"""
Flow frequency curve

Category: HydroSp
"""

import numpy as np


def hyfcr(flow=None, precip=None, coords=None, n=50):
    """Flow frequency curve

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


short = "hyfcr"
alias = "hyfcr"
quote = "No half measures. -- Mike"
hyfcr = hyfcr


def cheatsheet() -> str:
    return "hyfcr({}) -> Flow frequency curve"
