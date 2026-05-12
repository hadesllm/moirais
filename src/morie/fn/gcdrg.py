# morie.fn -- function file (hadesllm/morie)
"""
Drought trend spatial

Category: GeoClim
"""

import numpy as np


def gcdrg(data=None, coords=None, n=50):
    """Drought trend spatial

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


short = "gcdrg"
alias = "gcdrg"
quote = "Not all those who wander are lost. -- Gandalf"
gcdrg = gcdrg


def cheatsheet() -> str:
    return "gcdrg({}) -> Drought trend spatial"
