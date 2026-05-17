# morie.fn -- function file (hadesllm/morie)
"""
Spatial diffusion model

Category: SpatEpi2
"""

import numpy as np


def sespd(cases=None, population=None, coords=None, n=50):
    """Spatial diffusion model

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


short = "sespd"
alias = "sespd"
quote = "In the midst of chaos, there is also opportunity. -- Sun Tzu"
sespd = sespd


def cheatsheet() -> str:
    return "sespd({}) -> Spatial diffusion model"
