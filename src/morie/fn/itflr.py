# morie.fn — function file (hadesllm/morie)
"""Item floor and ceiling effects."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_floor_ceiling(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Floor (pct at min) and ceiling (pct at max) per item.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, min_val, max_val, floor_pct, ceiling_pct.

    References
    ----------
    McHorney, C. A. & Tarlov, A. R. (1995). Individual-patient monitoring
    in clinical practice. Medical Care, 33(1), AS77-AS88.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        nv = len(valid)
        if nv == 0:
            rows.append(
                {"item": names[j], "min_val": np.nan, "max_val": np.nan, "floor_pct": np.nan, "ceiling_pct": np.nan}
            )
            continue
        mn, mx = float(valid.min()), float(valid.max())
        floor_pct = float(np.sum(valid == mn) / nv * 100)
        ceil_pct = float(np.sum(valid == mx) / nv * 100)
        rows.append(
            {
                "item": names[j],
                "min_val": mn,
                "max_val": mx,
                "floor_pct": floor_pct,
                "ceiling_pct": ceil_pct,
            }
        )
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_floor_ceiling({}) -> Item floor and ceiling effects."
