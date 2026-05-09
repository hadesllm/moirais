# moirais.fn — function file (hadesllm/moirais)
"""
Phosphorus loading spatial

Category: HydroSp
"""

import numpy as np


def hyphs(flow=None, precip=None, coords=None, n=50):
    """Phosphorus loading spatial

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


short = "hyphs"
alias = "hyphs"
quote = "A Lannister always pays his debts. -- Tyrion"
hyphs = hyphs


def cheatsheet() -> str:
    return "hyphs({}) -> Phosphorus loading spatial"
