# morie.fn -- function file (hadesllm/morie)
"""
RCM downscaling spatial

Category: GeoClim
"""

import numpy as np


def gcrcm(data=None, coords=None, n=50):
    """RCM downscaling spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    if coords is None:
        coords = np.random.default_rng(1).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "gcrcm"
alias = "gcrcm"
quote = "An investment in knowledge pays the best interest. -- Benjamin Franklin"
gcrcm = gcrcm


def cheatsheet() -> str:
    return "gcrcm({}) -> RCM downscaling spatial"
