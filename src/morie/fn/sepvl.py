# morie.fn -- function file (hadesllm/morie)
"""
Prevalence spatial modeling

Category: SpatEpi2
"""

import numpy as np


def sepvl(cases=None, population=None, coords=None, n=50):
    """Prevalence spatial modeling

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


short = "sepvl"
alias = "sepvl"
quote = "A Lannister always pays his debts. -- Tyrion"
sepvl = sepvl


def cheatsheet() -> str:
    return "sepvl({}) -> Prevalence spatial modeling"
