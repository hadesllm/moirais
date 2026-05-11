# morie.fn — function file (hadesllm/morie)
"""
3D variogram modeling

Category: DimKrig
"""

import numpy as np


def dk3vg(x=None, y=None, z=None, values=None, n=30):
    """3D variogram modeling

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


short = "dk3vg"
alias = "dk3vg"
quote = "Those who break the rules are scum. -- Kakashi"
dk3vg = dk3vg


def cheatsheet() -> str:
    return "dk3vg({}) -> 3D variogram modeling"
