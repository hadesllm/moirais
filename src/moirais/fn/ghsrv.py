# moirais.fn — function file (hadesllm/moirais)
"""
Survival spatial health

Category: GeoHlth
"""

import numpy as np


def ghsrv(cases=None, controls=None, exposure=None, coords=None, n=50):
    """Survival spatial health

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


short = "ghsrv"
alias = "ghsrv"
quote = "My precious. -- Gollum"
ghsrv = ghsrv


def cheatsheet() -> str:
    return "ghsrv({}) -> Survival spatial health"
