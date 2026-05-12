# morie.fn -- function file (hadesllm/morie)
"""
Max coverage health

Category: GeoHlth
"""

import numpy as np


def ghmxc(cases=None, controls=None, exposure=None, coords=None, n=50):
    """Max coverage health

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if cases is None:
        cases = np.random.default_rng(0).poisson(10, n)
    if controls is None:
        controls = np.random.default_rng(1).poisson(100, n) + 10
    if exposure is None:
        exposure = np.random.default_rng(2).uniform(0, 1, n)
    if coords is None:
        coords = np.random.default_rng(3).uniform(0, 100, (n, 2))
    odds = cases / (controls + 1e-10)
    stat = float(np.mean(odds))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n": len(cases),
            "total_cases": int(np.sum(cases)),
            "mean_exposure": float(np.mean(exposure)),
            "mean_odds": float(np.mean(odds)),
        },
    )


short = "ghmxc"
alias = "ghmxc"
quote = "Keep moving forward. -- Eren"
ghmxc = ghmxc


def cheatsheet() -> str:
    return "ghmxc({}) -> Max coverage health"
