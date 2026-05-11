# morie.fn — function file (hadesllm/morie)
"""
Mortality atlas construction

Category: SpatEpi2
"""

import numpy as np


def semtl(cases=None, population=None, coords=None, n=50):
    """Mortality atlas construction

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


short = "semtl"
alias = "semtl"
quote = "Growing old is a blessing. -- Rengoku"
semtl = semtl


def cheatsheet() -> str:
    return "semtl({}) -> Mortality atlas construction"
