"""
Porosity soil spatial

Category: SoilSp
"""

import numpy as np


def sopor(data=None, depth=None, coords=None, n=50):
    """Porosity soil spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if data is None:
        data = np.random.default_rng(0).uniform(0, 1, n)
    if depth is None:
        depth = np.random.default_rng(1).uniform(0, 2, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "mean_depth": float(np.mean(depth))},
    )


short = "sopor"
alias = "sopor"
quote = "I am the one who knocks. -- Walter White"
sopor = sopor


def cheatsheet() -> str:
    return "sopor({}) -> Porosity soil spatial"
