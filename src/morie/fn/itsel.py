# morie.fn — function file (hadesllm/morie)
"""Item selection — flag items for removal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_select(
    data: pd.DataFrame | np.ndarray,
    *,
    min_r: float = 0.3,
    max_floor: float = 0.80,
    max_ceiling: float = 0.80,
) -> pd.DataFrame:
    """Flag items that should be considered for removal.

    Criteria:
    - Corrected item-total r < min_r (low discrimination)
    - Floor pct > max_floor (too easy / extreme floor)
    - Ceiling pct > max_ceiling (too easy / extreme ceiling)

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    min_r : float
        Minimum corrected item-total correlation (default 0.3).
    max_floor : float
        Maximum acceptable floor percentage as proportion (default 0.80).
    max_ceiling : float
        Maximum acceptable ceiling percentage as proportion (default 0.80).

    Returns
    -------
    DataFrame
        Columns: item, r_corrected, floor_pct, ceiling_pct, flag_r,
        flag_floor, flag_ceiling, flagged.

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
        rest = total - col
        sd_i = np.std(col, ddof=1)
        sd_r = np.std(rest, ddof=1)
        if sd_i < 1e-15 or sd_r < 1e-15:
            rc = 0.0
        else:
            rc = float(np.corrcoef(col, rest)[0, 1])

        valid = col[~np.isnan(col)]
        nv = len(valid)
        if nv > 0:
            mn, mx = valid.min(), valid.max()
            floor_pct = float(np.sum(valid == mn) / nv)
            ceil_pct = float(np.sum(valid == mx) / nv)
        else:
            floor_pct = ceil_pct = np.nan

        flag_r = rc < min_r
        flag_f = floor_pct > max_floor if not np.isnan(floor_pct) else False
        flag_c = ceil_pct > max_ceiling if not np.isnan(ceil_pct) else False
        rows.append(
            {
                "item": names[j],
                "r_corrected": rc,
                "floor_pct": floor_pct,
                "ceiling_pct": ceil_pct,
                "flag_r": flag_r,
                "flag_floor": flag_f,
                "flag_ceiling": flag_c,
                "flagged": flag_r or flag_f or flag_c,
            }
        )
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_select({}) -> Item selection — flag items for removal."
