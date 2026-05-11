# morie.fn — function file (hadesllm/morie)
"""
3D ordinary kriging

Category: DimKrig
"""

import numpy as np


def dk3ok(x=None, y=None, z=None, values=None, n=30):
    """3D ordinary kriging

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


short = "dk3ok"
alias = "dk3ok"
quote = "Walk without rhythm. -- Fremen proverb"
dk3ok = dk3ok


def cheatsheet() -> str:
    return "dk3ok({}) -> 3D ordinary kriging"
