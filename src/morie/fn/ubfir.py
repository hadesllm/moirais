"""
Fire station coverage

Category: UrbanSp
"""

import numpy as np


def ubfir(population=None, area=None, coords=None, n=50):
    """Fire station coverage

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


short = "ubfir"
alias = "ubfir"
quote = "Total Concentration Breathing. -- Tanjiro"
ubfir = ubfir


def cheatsheet() -> str:
    return "ubfir({}) -> Fire station coverage"
