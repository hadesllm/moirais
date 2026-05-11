# morie.fn — function file (hadesllm/morie)
"""
Polar vortex spatial

Category: GeoClim
"""

import numpy as np


def gcplr(data=None, coords=None, n=50):
    """Polar vortex spatial

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


short = "gcplr"
alias = "gcplr"
quote = "See you space cowboy. -- Spike"
gcplr = gcplr


def cheatsheet() -> str:
    return "gcplr({}) -> Polar vortex spatial"
