# morie.fn -- function file (hadesllm/morie)
"""Complete (single-year) life table construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import ESRes


def life_table_complete(
    ages: list[int] | np.ndarray,
    deaths: list[int] | np.ndarray,
    populations: list[int] | np.ndarray,
    radix: int = 100000,
) -> ESRes:
    """Construct a complete (single-year) cohort life table.

    Unlike ``ltab.py`` (abridged), this constructs a full single-year
    life table with all standard columns: mx, qx, lx, dx, Lx, Tx, ex.

    Parameters
    ----------
    ages : array-like of int
        Age values (0, 1, 2, ...).
    deaths : array-like of int
        Deaths at each age.
    populations : array-like of int
        Mid-year population at each age.
    radix : int, default 100000
        Starting cohort size (l0).

    Returns
    -------
    ESRes
        extra contains a DataFrame with all life table columns.

    References
    ----------
    Preston, S. H. et al. (2001). *Demography: Measuring and Modeling
    Population Processes*. Blackwell, Ch. 3.
    """
    a = np.asarray(ages, dtype=int)
    d = np.asarray(deaths, dtype=float)
    n = np.asarray(populations, dtype=float)

    if len(a) != len(d) or len(a) != len(n):
        raise ValueError("All arrays must have the same length")

    k = len(a)
    mx = d / n
    qx = np.zeros(k)
    for i in range(k - 1):
        qx[i] = mx[i] / (1 + 0.5 * mx[i])
    qx[-1] = 1.0

    lx = np.zeros(k)
    lx[0] = radix
    dx = np.zeros(k)
    for i in range(k - 1):
        dx[i] = lx[i] * qx[i]
        lx[i + 1] = lx[i] - dx[i]
    dx[-1] = lx[-1]

    Lx = np.zeros(k)
    for i in range(k - 1):
        Lx[i] = lx[i + 1] + 0.5 * dx[i]
    Lx[-1] = lx[-1] / mx[-1] if mx[-1] > 0 else 0.0

    Tx = np.zeros(k)
    Tx[-1] = Lx[-1]
    for i in range(k - 2, -1, -1):
        Tx[i] = Tx[i + 1] + Lx[i]

    ex = np.zeros(k)
    for i in range(k):
        ex[i] = Tx[i] / lx[i] if lx[i] > 0 else 0.0

    df = pd.DataFrame({
        "age": a, "mx": mx, "qx": qx, "lx": lx,
        "dx": dx, "Lx": Lx, "Tx": Tx, "ex": ex,
    })

    return ESRes(
        measure="life_table_complete",
        estimate=float(ex[0]),
        n=int(np.sum(d)),
        extra={"table": df, "radix": radix, "e0": float(ex[0])},
    )


lftbl = life_table_complete


def cheatsheet() -> str:
    return "life_table_complete({}) -> Complete single-year life table."
