# morie.fn -- function file (hadesllm/morie)
"""Life table mortality rates qx."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def life_table_qx(
    deaths: np.ndarray,
    population: np.ndarray,
    ages: np.ndarray | None = None,
) -> DescriptiveResult:
    """Compute age-specific mortality rates qx from a life table.

    The probability of dying in the age interval [x, x+1) is:

    .. math::

        q_x = \\frac{d_x}{l_x}

    where :math:`d_x` is deaths and :math:`l_x` is the population
    alive at exact age *x*.  Life expectancy at each age is computed
    from the resulting survivorship function.

    Parameters
    ----------
    deaths : array-like
        Number of deaths in each age group.
    population : array-like
        Mid-interval (or start-of-interval) population in each group.
    ages : array-like, optional
        Age labels.  If None, uses 0, 1, 2, ...

    Returns
    -------
    DescriptiveResult
        ``value`` is the life expectancy at birth (e0).
        ``extra`` has ``qx``, ``lx``, ``dx``, ``Lx``, ``Tx``, ``ex``,
        ``ages``.

    Raises
    ------
    ValueError
        If array lengths mismatch or non-positive populations.

    References
    ----------
    Preston, S. H., Heuveline, P., & Guillot, M. (2001).
    *Demography: Measuring and Modeling Population Processes*.
    Blackwell.
    """
    d = np.asarray(deaths, dtype=np.float64).ravel()
    pop = np.asarray(population, dtype=np.float64).ravel()
    if len(d) != len(pop):
        raise ValueError("deaths and population must have same length.")
    if np.any(pop <= 0):
        raise ValueError("All population values must be positive.")

    n = len(d)
    if ages is None:
        ages_arr = np.arange(n)
    else:
        ages_arr = np.asarray(ages).ravel()
        if len(ages_arr) != n:
            raise ValueError("ages must match deaths/population length.")

    mx = d / pop
    qx = np.zeros(n)
    for i in range(n - 1):
        qx[i] = 1 - np.exp(-mx[i])
    qx[-1] = 1.0

    radix = 100000.0
    lx = np.zeros(n)
    lx[0] = radix
    dx = np.zeros(n)
    for i in range(n):
        dx[i] = lx[i] * qx[i]
        if i < n - 1:
            lx[i + 1] = lx[i] - dx[i]

    Lx = np.zeros(n)
    for i in range(n - 1):
        Lx[i] = (lx[i] + lx[i + 1]) / 2.0
    Lx[-1] = lx[-1] / mx[-1] if mx[-1] > 0 else 0.0

    Tx = np.zeros(n)
    Tx[-1] = Lx[-1]
    for i in range(n - 2, -1, -1):
        Tx[i] = Tx[i + 1] + Lx[i]

    ex = np.zeros(n)
    for i in range(n):
        ex[i] = Tx[i] / lx[i] if lx[i] > 0 else 0.0

    return DescriptiveResult(
        name="LifeTable",
        value=float(ex[0]),
        extra={
            "qx": qx,
            "lx": lx,
            "dx": dx,
            "Lx": Lx,
            "Tx": Tx,
            "ex": ex,
            "ages": ages_arr,
        },
    )


lifeq = life_table_qx


def cheatsheet() -> str:
    return "life_table_qx({}) -> Life table mortality rates qx."
