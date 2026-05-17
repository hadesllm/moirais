# morie.fn -- function file (hadesllm/morie)
"""
D8 flow direction

Category: HydroSp
"""

import numpy as np


def hyd8f(flow=None, precip=None, coords=None, n=50):
    """D8 flow direction

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


short = "hyd8f"
alias = "hyd8f"
quote = "I think, therefore I am. -- Rene Descartes"
hyd8f = hyd8f


def cheatsheet() -> str:
    return "hyd8f({}) -> D8 flow direction"
