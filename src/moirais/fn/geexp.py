# moirais.fn — function file (hadesllm/moirais)
"""
Export spatial mapping

Category: GeoEcon
"""

import numpy as np


def geexp(gdp=None, trade=None, coords=None, n=50):
    """Export spatial mapping

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if gdp is None:
        gdp = np.random.default_rng(0).uniform(1000, 100000, n)
    if trade is None:
        trade = np.random.default_rng(1).uniform(100, 50000, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(-180, 180, (n, 2))
    stat = float(np.mean(gdp))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(gdp), "mean_gdp": float(np.mean(gdp)), "total_trade": float(np.sum(trade))},
    )


short = "geexp"
alias = "geexp"
quote = "This is Requiem. -- Giorno"
geexp = geexp


def cheatsheet() -> str:
    return "geexp({}) -> Export spatial mapping"
