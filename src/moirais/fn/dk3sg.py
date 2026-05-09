# moirais.fn — function file (hadesllm/moirais)
"""
3D sequential Gaussian sim

Category: DimKrig
"""

import numpy as np


def dk3sg(x=None, y=None, z=None, values=None, n=30):
    """3D sequential Gaussian sim

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


short = "dk3sg"
alias = "dk3sg"
quote = "I will take a potato chip and eat it! -- Light"
dk3sg = dk3sg


def cheatsheet() -> str:
    return "dk3sg({}) -> 3D sequential Gaussian sim"
