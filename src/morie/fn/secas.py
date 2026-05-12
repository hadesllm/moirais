# morie.fn -- function file (hadesllm/morie)
"""
Case-control spatial study

Category: SpatEpi2
"""

import numpy as np


def secas(cases=None, population=None, coords=None, n=50):
    """Case-control spatial study

    Returns
    -------
    DescriptiveResult
    """
    from ._containers import DescriptiveResult

    if cases is None:
        cases = np.random.default_rng(0).poisson(5, n)
    if population is None:
        population = np.random.default_rng(1).poisson(1000, n) + 100
    if coords is None:
        coords = np.random.default_rng(2).uniform(0, 100, (n, 2))
    rates = cases / population
    stat = float(np.mean(rates))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={
            "n_areas": len(cases),
            "total_cases": int(np.sum(cases)),
            "total_pop": int(np.sum(population)),
            "mean_rate": float(np.mean(rates)),
        },
    )


short = "secas"
alias = "secas"
quote = "Walk without rhythm. -- Fremen proverb"
secas = secas


def cheatsheet() -> str:
    return "secas({}) -> Case-control spatial study"
