# morie.fn -- function file (hadesllm/morie)
"""
Total suspended solids

Category: HydroSp
"""

import numpy as np


def hytss(flow=None, precip=None, coords=None, n=50):
    """Total suspended solids

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


short = "hytss"
alias = "hytss"
quote = "What is now proved was once only imagined. -- William Blake"
hytss = hytss


def cheatsheet() -> str:
    return "hytss({}) -> Total suspended solids"
