# morie.fn -- function file (hadesllm/morie)
"""
Upwelling change spatial

Category: GeoClim
"""

import numpy as np


def gcupw(data=None, coords=None, n=50):
    """Upwelling change spatial

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


short = "gcupw"
alias = "gcupw"
quote = "The only true wisdom is in knowing you know nothing. -- Socrates"
gcupw = gcupw


def cheatsheet() -> str:
    return "gcupw({}) -> Upwelling change spatial"
