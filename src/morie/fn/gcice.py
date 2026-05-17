# morie.fn -- function file (hadesllm/morie)
"""
Ice sheet change spatial

Category: GeoClim
"""

import numpy as np


def gcice(data=None, coords=None, n=50):
    """Ice sheet change spatial

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


short = "gcice"
alias = "gcice"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
gcice = gcice


def cheatsheet() -> str:
    return "gcice({}) -> Ice sheet change spatial"
