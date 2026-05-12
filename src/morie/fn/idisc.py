# morie.fn -- function file (hadesllm/morie)
"""Item discrimination index (D-statistic)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def idisc(
    data: pd.DataFrame | np.ndarray,
    pct: float = 0.27,
) -> pd.DataFrame:
    """Item discrimination index (D-statistic).

    Compares upper and lower groups by total score using Kelley's
    recommended 27 percent split.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    pct : float
        Proportion for upper/lower groups (default 0.27).

    Returns
    -------
    DataFrame
        Columns: item, d (discrimination index).
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    total = X.sum(axis=1)
    cut = max(int(n * pct), 1)
    si = np.argsort(total)
    lo, hi = si[:cut], si[-cut:]

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        mx = X[:, j].max()
        pu = X[hi, j].mean() / mx if mx > 0 else 0
        pl = X[lo, j].mean() / mx if mx > 0 else 0
        rows.append({"item": names[j], "d": float(pu - pl)})
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "idisc({}) -> Item discrimination index (D-statistic)."
