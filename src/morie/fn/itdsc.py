# morie.fn — function file (hadesllm/morie)
"""Item discrimination for all items (corrected item-total r)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_discrimination_all(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Corrected item-total correlation for every item.

    The corrected item-total correlation removes the item from the total
    before computing the Pearson r, avoiding part-whole inflation.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, r_corrected.

    References
    ----------
    Nunnally, J. C. & Bernstein, I. H. (1994). Psychometric Theory (3rd ed.).
    McGraw-Hill.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    total = X.sum(axis=1)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        rest = total - X[:, j]
        sd_item = np.std(X[:, j], ddof=1)
        sd_rest = np.std(rest, ddof=1)
        if sd_item < 1e-15 or sd_rest < 1e-15:
            rc = 0.0
        else:
            rc = float(np.corrcoef(X[:, j], rest)[0, 1])
        rows.append({"item": names[j], "r_corrected": rc})
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_discrimination_all({}) -> Item discrimination for all items (corrected item-total r)."
