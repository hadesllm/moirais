# morie.fn -- function file (hadesllm/morie)
"""Abridged life table construction."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def life_table(
    age_starts: Union[list, np.ndarray],
    deaths: Union[list, np.ndarray],
    population: Union[list, np.ndarray],
    *,
    interval_widths: Union[list, np.ndarray, None] = None,
    radix: int = 100000,
) -> dict[str, Any]:
    """
    Construct an abridged period life table.

    Columns computed:
      * nMx -- age-specific death rate
      * nqx -- probability of dying in interval
      * lx  -- number surviving to age x (starting from radix)
      * ndx -- number dying in interval
      * nLx -- person-years lived in interval
      * Tx  -- total person-years lived above age x
      * ex  -- life expectancy at age x

    Conversion from nMx to nqx uses: nqx = n * nMx / (1 + (n - nax) * nMx)
    with nax = n/2 (mid-interval assumption) except for the first interval
    where nax = 0.1 for infants.

    :param age_starts: Start of each age interval (e.g., [0, 1, 5, 10, ...]).
    :param deaths: Deaths in each interval.
    :param population: Mid-period population in each interval.
    :param interval_widths: Width of each interval. If None, inferred from
        age_starts (last interval assumed width 5).
    :param radix: Starting population (default 100000).
    :return: Dictionary with arrays: age, nMx, nqx, lx, ndx, nLx, Tx, ex.
    :raises ValueError: If arrays have different lengths.

    References
    ----------
    Preston, S. H., Heuveline, P., & Guillot, M. (2001). *Demography:
    Measuring and Modeling Population Processes*. Blackwell, Ch. 3.
    """
    a = np.asarray(age_starts, dtype=float)
    d = np.asarray(deaths, dtype=float)
    N = np.asarray(population, dtype=float)
    k = len(a)
    if len(d) != k or len(N) != k:
        raise ValueError("age_starts, deaths, population must have the same length.")

    if interval_widths is not None:
        n = np.asarray(interval_widths, dtype=float)
    else:
        n = np.diff(a, append=a[-1] + 5) if k > 1 else np.array([5.0])

    # nMx
    nMx = d / np.maximum(N, 1e-12)

    # nax: fraction of interval lived by those who die
    nax = n / 2.0
    if k > 0:
        nax[0] = 0.1  # infant approximation

    # nqx
    nqx = n * nMx / (1.0 + (n - nax) * nMx)
    nqx = np.clip(nqx, 0.0, 1.0)
    nqx[-1] = 1.0  # last interval: everyone dies

    # lx, ndx
    lx = np.empty(k)
    ndx = np.empty(k)
    lx[0] = radix
    for i in range(k):
        ndx[i] = lx[i] * nqx[i]
        if i + 1 < k:
            lx[i + 1] = lx[i] - ndx[i]

    # nLx
    nLx = np.empty(k)
    for i in range(k - 1):
        nLx[i] = n[i] * lx[i + 1] + nax[i] * ndx[i] if i + 1 < k else n[i] * lx[i]
    # Last interval: nLx = lx / nMx (stationary population)
    nLx[-1] = lx[-1] / nMx[-1] if nMx[-1] > 0 else 0.0

    # Tx, ex
    Tx = np.cumsum(nLx[::-1])[::-1]
    ex = Tx / np.maximum(lx, 1e-12)

    return {
        "age": a.tolist(),
        "nMx": nMx.tolist(),
        "nqx": nqx.tolist(),
        "lx": lx.tolist(),
        "ndx": ndx.tolist(),
        "nLx": nLx.tolist(),
        "Tx": Tx.tolist(),
        "ex": ex.tolist(),
    }


ltab = life_table


def cheatsheet() -> str:
    return "life_table({}) -> Abridged life table construction."
