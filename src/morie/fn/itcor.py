# morie.fn — function file (hadesllm/morie)
"""Corrected item-total correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def itcor(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Corrected item-total correlations.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, r_total (uncorrected), r_corr (corrected).
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    total = X.sum(axis=1)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]

    rows = []
    for j in range(k):
        item = X[:, j]
        rt = float(np.corrcoef(item, total)[0, 1])
        rc = float(np.corrcoef(item, total - item)[0, 1])
        rows.append({"item": names[j], "r_total": rt, "r_corr": rc})
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "itcor({}) -> Corrected item-total correlations."
