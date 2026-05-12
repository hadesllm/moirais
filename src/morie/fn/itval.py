# morie.fn -- function file (hadesllm/morie)
"""Item validity index."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_validity_index(
    data: pd.DataFrame | np.ndarray,
    criterion: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Validity index = r_ic * SD_item for each item.

    Measures each item's contribution to criterion prediction.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    criterion : array-like
        External criterion variable (length n).

    Returns
    -------
    DataFrame
        Columns: item, r_criterion, sd, validity_index.

    References
    ----------
    Ebel, R. L. & Frisbie, D. A. (1991). Essentials of Educational
    Measurement (5th ed.). Prentice Hall.
    """
    X = np.asarray(data, dtype=np.float64)
    c = np.asarray(criterion, dtype=np.float64).ravel()
    n, k = X.shape
    if len(c) != n:
        raise ValueError(f"criterion length {len(c)} != n_respondents {n}")

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    sd_c = float(np.std(c, ddof=1))
    rows = []
    for j in range(k):
        sd_item = float(np.std(X[:, j], ddof=1))
        if sd_item < 1e-15 or sd_c < 1e-15:
            ric = 0.0
        else:
            ric = float(np.corrcoef(X[:, j], c)[0, 1])
        vi = ric * sd_item
        rows.append(
            {
                "item": names[j],
                "r_criterion": ric,
                "sd": sd_item,
                "validity_index": vi,
            }
        )
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_validity_index({}) -> Item validity index."
