# morie.fn -- function file (hadesllm/morie)
"""
Wealth spatial mapping

Category: GeoEcon
"""

import numpy as np


def gewlt(gdp=None, trade=None, coords=None, n=50):
    """Wealth spatial mapping

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


short = "gewlt"
alias = "gewlt"
quote = "Number rules the universe. -- Pythagoras"
gewlt = gewlt


def cheatsheet() -> str:
    return "gewlt({}) -> Wealth spatial mapping"
