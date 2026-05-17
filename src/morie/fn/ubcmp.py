"""
Urban compactness

Category: UrbanSp
"""

import numpy as np


def ubcmp(population=None, area=None, coords=None, n=50):
    """Urban compactness

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if population is None:
        population = np.random.default_rng(0).poisson(5000, n)
    if area is None:
        area = np.random.default_rng(1).uniform(1, 100, n)
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    density = population / area
    stat = float(np.mean(density))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(population), "mean_density": float(np.mean(density)), "total_pop": int(np.sum(population))},
    )


short = "ubcmp"
alias = "ubcmp"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
ubcmp = ubcmp


def cheatsheet() -> str:
    return "ubcmp({}) -> Urban compactness"
