# morie.fn -- function file (hadesllm/morie)
"""Life expectancy from an abridged life table."""

import numpy as np

from ._containers import DescriptiveResult


def life_expectancy(
    age_starts: np.ndarray,
    deaths: np.ndarray,
    population: np.ndarray,
    ax_fraction: float = 0.5,
) -> DescriptiveResult:
    """Life expectancy at birth from an abridged life table.

    Constructs columns: nMx, nqx, lx, ndx, nLx, Tx, ex.

    Parameters
    ----------
    age_starts : array-like
        Lower bound of each age interval (e.g. [0, 1, 5, 10, ...]).
    deaths : array-like
        Deaths in each age interval.
    population : array-like
        Mid-year population in each age interval.
    ax_fraction : float, default 0.5
        Average fraction of interval lived by those dying (nax/n).

    Returns
    -------
    DescriptiveResult
        value = life expectancy at birth (e0).
        extra contains full life table arrays.

    References
    ----------
    Preston, S. H., Heuveline, P., & Guillot, M. (2001). Demography:
    Measuring and Modeling Population Processes. Blackwell.
    """
    age_starts = np.asarray(age_starts, dtype=float)
    deaths = np.asarray(deaths, dtype=float)
    population = np.asarray(population, dtype=float)
    k = len(age_starts)

    if k < 2:
        raise ValueError("Need at least 2 age groups")

    n = np.diff(age_starts, append=age_starts[-1] + 5)
    nMx = deaths / population
    nax = ax_fraction * n

    nqx = np.zeros(k)
    for i in range(k - 1):
        nqx[i] = (n[i] * nMx[i]) / (1 + (n[i] - nax[i]) * nMx[i])
    nqx[-1] = 1.0
    nqx = np.clip(nqx, 0, 1)

    lx = np.zeros(k)
    lx[0] = 100_000.0
    for i in range(1, k):
        lx[i] = lx[i - 1] * (1 - nqx[i - 1])

    ndx = lx * nqx

    nLx = np.zeros(k)
    for i in range(k - 1):
        nLx[i] = n[i] * lx[i + 1] + nax[i] * ndx[i]
    nLx[-1] = lx[-1] / nMx[-1] if nMx[-1] > 0 else 0.0

    Tx = np.cumsum(nLx[::-1])[::-1]
    ex = Tx / lx
    ex = np.where(np.isfinite(ex), ex, 0.0)

    return DescriptiveResult(
        name="Life expectancy",
        value=float(ex[0]),
        extra={
            "age_starts": age_starts,
            "nMx": nMx,
            "nqx": nqx,
            "lx": lx,
            "ndx": ndx,
            "nLx": nLx,
            "Tx": Tx,
            "ex": ex,
        },
    )


le = life_expectancy


def cheatsheet() -> str:
    return "life_expectancy({}) -> Life expectancy from an abridged life table."
