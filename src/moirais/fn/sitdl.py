"""Per-item detail within a subscale."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._mapq_const import SUBSCALES
from moirais.fn.crba import crba


def subscale_item_detail(
    data: pd.DataFrame,
    subscale_name: str,
    *,
    items: list[str] | None = None,
) -> pd.DataFrame:
    """Per-item statistics within a subscale.

    Parameters
    ----------
    data : DataFrame
        Item response data.
    subscale_name : str
        Subscale key (e.g., 'EE', 'EA', 'UA', 'ER') or any custom name
        if items are provided.
    items : list of str, optional
        Item column names. Default: looked up from SUBSCALES.

    Returns
    -------
    DataFrame
        Columns: item, mean, sd, skew, corrected_item_total_r,
        alpha_if_deleted.

    References
    ----------
    Nunnally, J.C. & Bernstein, I.H. (1994). Psychometric Theory.
        McGraw-Hill.
    """
    if items is not None:
        cols = items
    elif subscale_name in SUBSCALES:
        cols = SUBSCALES[subscale_name]
    else:
        raise ValueError(f"Unknown subscale: {subscale_name!r}. Provide items= or use one of {list(SUBSCALES.keys())}.")

    cols = [c for c in cols if c in data.columns]
    if len(cols) < 2:
        raise ValueError(f"Need >= 2 items, got {len(cols)}")

    X = data[cols].dropna()
    arr = X.to_numpy(dtype=np.float64)
    total = arr.sum(axis=1)

    rows = []
    for j, col in enumerate(cols):
        item_vals = arr[:, j]
        # Corrected item-total correlation (remove item from total)
        rest_total = total - item_vals
        r_it = float(np.corrcoef(item_vals, rest_total)[0, 1])

        # Alpha if deleted
        mask = list(range(len(cols)))
        mask.remove(j)
        alpha_del = crba(arr[:, mask]).raw

        rows.append(
            {
                "item": col,
                "mean": float(np.mean(item_vals)),
                "sd": float(np.std(item_vals, ddof=1)),
                "skew": float(sp.skew(item_vals)),
                "corrected_item_total_r": r_it,
                "alpha_if_deleted": alpha_del,
            }
        )

    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "subscale_item_detail({}) -> Per-item detail within a subscale."
