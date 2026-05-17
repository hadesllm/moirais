# morie.fn -- function file (hadesllm/morie)
"""
SDM health

Category: GeoHlth
"""

import numpy as np


def ghsdm(cases=None, controls=None, exposure=None, coords=None, n=50):
    """SDM health

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


short = "ghsdm"
alias = "ghsdm"
quote = "Statistics is the grammar of science. -- Karl Pearson"
ghsdm = ghsdm


def cheatsheet() -> str:
    return "ghsdm({}) -> SDM health"
