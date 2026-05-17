# morie.fn -- function file (hadesllm/morie)
"""
Risk mapping health

Category: GeoHlth
"""

import numpy as np


def ghrsk(cases=None, controls=None, exposure=None, coords=None, n=50):
    """Risk mapping health

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


short = "ghrsk"
alias = "ghrsk"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
ghrsk = ghrsk


def cheatsheet() -> str:
    return "ghrsk({}) -> Risk mapping health"
