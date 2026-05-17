# morie.fn -- function file (hadesllm/morie)
"""
GCM downscaling spatial

Category: GeoClim
"""

import numpy as np


def gcgcm(data=None, coords=None, n=50):
    """GCM downscaling spatial

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


short = "gcgcm"
alias = "gcgcm"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
gcgcm = gcgcm


def cheatsheet() -> str:
    return "gcgcm({}) -> GCM downscaling spatial"
