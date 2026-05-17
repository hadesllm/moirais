# morie.fn -- function file (hadesllm/morie)
"""
Family size spatial

Category: GeoDem
"""

import numpy as np


def gdflm(population=None, births=None, deaths=None, coords=None, n=50):
    """Family size spatial

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if population is None:
        population = np.random.default_rng(0).poisson(10000, n)
    if births is None:
        births = np.random.default_rng(1).poisson(100, n)
    if deaths is None:
        deaths = np.random.default_rng(2).poisson(80, n)
    if coords is None:
        coords = np.random.default_rng(3).uniform(0, 100, (n, 2))
    growth_rate = (births - deaths) / population
    stat = float(np.mean(growth_rate))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(population), "total_pop": int(np.sum(population)), "mean_growth": float(np.mean(growth_rate))},
    )


short = "gdflm"
alias = "gdflm"
quote = "An investment in knowledge pays the best interest. -- Benjamin Franklin"
gdflm = gdflm


def cheatsheet() -> str:
    return "gdflm({}) -> Family size spatial"
