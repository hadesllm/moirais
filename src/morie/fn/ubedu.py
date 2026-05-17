"""
Education accessibility

Category: UrbanSp
"""

import numpy as np


def ubedu(population=None, area=None, coords=None, n=50):
    """Education accessibility

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


short = "ubedu"
alias = "ubedu"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
ubedu = ubedu


def cheatsheet() -> str:
    return "ubedu({}) -> Education accessibility"
