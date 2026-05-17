# morie.fn -- function file (hadesllm/morie)
"""
Moving window scan

Category: SpatEpi2
"""

import numpy as np


def seclm(cases=None, population=None, coords=None, n=50):
    """Moving window scan

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


short = "seclm"
alias = "seclm"
quote = "The measure of a man is what he does with power. -- Plato"
seclm = seclm


def cheatsheet() -> str:
    return "seclm({}) -> Moving window scan"
