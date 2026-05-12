# morie.fn -- function file (hadesllm/morie)
"""Full item analysis table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn.crba import crba


def item_table(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Full item analysis table combining all key item statistics.

    Columns: item, mean, sd, difficulty, r_corrected, alpha_if_deleted,
    floor_pct, ceiling_pct, skewness.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Comprehensive item analysis table.

    References
    ----------
    DeVellis, R. F. (2017). Scale Development: Theory and Applications
    (4th ed.). SAGE.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    total = X.sum(axis=1)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        nv = len(valid)
        mean_j = float(np.mean(valid)) if nv > 0 else np.nan
        sd_j = float(np.std(valid, ddof=1)) if nv > 1 else np.nan
        mx = float(valid.max()) if nv > 0 else np.nan
        diff = mean_j / mx if mx > 0 else np.nan

        rest = total - col
        sd_r = np.std(rest, ddof=1) if nv > 1 else 0.0
        if sd_j is not np.nan and sd_j > 1e-15 and sd_r > 1e-15:
            rc = float(np.corrcoef(col, rest)[0, 1])
        else:
            rc = 0.0

        a_del = crba(np.delete(X, j, axis=1)).raw if k > 2 else np.nan

        if nv > 0:
            mn = float(valid.min())
            floor_pct = float(np.sum(valid == mn) / nv * 100)
            ceil_pct = float(np.sum(valid == mx) / nv * 100)
        else:
            floor_pct = ceil_pct = np.nan

        skew = float(sp.skew(valid, bias=False)) if nv >= 3 else np.nan

        rows.append(
            {
                "item": names[j],
                "mean": mean_j,
                "sd": sd_j,
                "difficulty": diff,
                "r_corrected": rc,
                "alpha_if_deleted": a_del,
                "floor_pct": floor_pct,
                "ceiling_pct": ceil_pct,
                "skewness": skew,
            }
        )
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_table({}) -> Full item analysis table."
