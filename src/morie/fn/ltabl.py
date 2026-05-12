# morie.fn -- function file (hadesllm/morie)
"""Full abridged life table."""

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def life_table_full(
    age_groups: list | np.ndarray,
    deaths: list | np.ndarray,
    populations: list | np.ndarray,
    interval: int = 5,
) -> DescriptiveResult:
    """Construct abridged life table.

    Parameters
    ----------
    age_groups : array-like
        Start of each age group.
    deaths : array-like
        Deaths in each group.
    populations : array-like
        Mid-year population in each group.
    interval : int
        Width of age intervals.

    Returns
    -------
    DescriptiveResult
    """
    ag = np.asarray(age_groups, dtype=float)
    d = np.asarray(deaths, dtype=float)
    p = np.asarray(populations, dtype=float)
    n = len(ag)

    if n < 2 or len(d) != n or len(p) != n:
        raise ValueError("All arrays must match and have >= 2 groups")

    mx = d / p
    ax = np.full(n, interval / 2)
    ax[0] = 0.1 if mx[0] >= 0.107 else 0.05 + 3.0 * mx[0]

    qx = np.zeros(n)
    for i in range(n - 1):
        qx[i] = (interval * mx[i]) / (1 + (interval - ax[i]) * mx[i])
    qx[-1] = 1.0
    qx = np.clip(qx, 0, 1)

    lx = np.zeros(n)
    lx[0] = 100000
    for i in range(1, n):
        lx[i] = lx[i - 1] * (1 - qx[i - 1])

    dx = lx * qx
    nLx = np.zeros(n)
    for i in range(n - 1):
        nLx[i] = interval * lx[i + 1] + ax[i] * dx[i]
    nLx[-1] = lx[-1] / mx[-1] if mx[-1] > 0 else 0

    Tx = np.flip(np.cumsum(np.flip(nLx)))
    ex = Tx / lx

    tbl = pd.DataFrame(
        {
            "age": ag,
            "mx": mx,
            "qx": qx,
            "lx": lx,
            "dx": dx,
            "nLx": nLx,
            "Tx": Tx,
            "ex": ex,
        }
    )

    return DescriptiveResult(
        name="life_table",
        value=tbl,
        extra={"e0": float(ex[0]), "n_groups": n},
    )


ltabl = life_table_full


def cheatsheet() -> str:
    return "life_table_full({}) -> Full abridged life table."
