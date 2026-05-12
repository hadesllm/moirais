# morie.fn -- function file (hadesllm/morie)
"""
Tax revenue spatial

Category: GeoEcon
"""

import numpy as np


def getax(gdp=None, trade=None, coords=None, n=50):
    """Tax revenue spatial

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


short = "getax"
alias = "getax"
quote = "My precious. -- Gollum"
getax = getax


def cheatsheet() -> str:
    return "getax({}) -> Tax revenue spatial"
