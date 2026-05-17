# morie.fn -- function file (hadesllm/morie)
"""
Vegetation shift spatial

Category: GeoClim
"""

import numpy as np


def gcveg(data=None, coords=None, n=50):
    """Vegetation shift spatial

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


short = "gcveg"
alias = "gcveg"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
gcveg = gcveg


def cheatsheet() -> str:
    return "gcveg({}) -> Vegetation shift spatial"
