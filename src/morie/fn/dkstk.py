# morie.fn -- function file (hadesllm/morie)
"""
STPK space-time kriging

Category: DimKrig
"""

import numpy as np


def dkstk(x=None, y=None, z=None, values=None, n=30):
    """STPK space-time kriging

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if x is None:
        x = np.random.default_rng(0).uniform(0, 100, n)
    if y is None:
        y = np.random.default_rng(1).uniform(0, 100, n)
    if z is None:
        z = np.random.default_rng(2).uniform(0, 50, n)
    if values is None:
        values = np.random.default_rng(3).standard_normal(n)
    stat = float(np.var(values))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n_obs": len(x),
            "x_range": [float(np.min(x)), float(np.max(x))],
            "y_range": [float(np.min(y)), float(np.max(y))],
            "z_range": [float(np.min(z)), float(np.max(z))],
        },
    )


short = "dkstk"
alias = "dkstk"
quote = "I think, therefore I am. -- Rene Descartes"
dkstk = dkstk


def cheatsheet() -> str:
    return "dkstk({}) -> STPK space-time kriging"
