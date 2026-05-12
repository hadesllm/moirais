# morie.fn -- function file (hadesllm/morie)
"""
Flood trend spatial

Category: GeoClim
"""

import numpy as np


def gcfld(data=None, coords=None, n=50):
    """Flood trend spatial

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


short = "gcfld"
alias = "gcfld"
quote = "Make it so. -- Picard"
gcfld = gcfld


def cheatsheet() -> str:
    return "gcfld({}) -> Flood trend spatial"
