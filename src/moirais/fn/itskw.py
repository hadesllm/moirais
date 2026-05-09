# moirais.fn — function file (hadesllm/moirais)
"""Item skewness and kurtosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp


def item_skew_kurt(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Skewness and excess kurtosis per item.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, skewness, kurtosis.

    References
    ----------
    Tabachnick, B. G. & Fidell, L. S. (2013). Using Multivariate Statistics
    (6th ed.). Pearson.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 3:
            rows.append({"item": names[j], "skewness": np.nan, "kurtosis": np.nan})
            continue
        rows.append(
            {
                "item": names[j],
                "skewness": float(sp.skew(valid, bias=False)),
                "kurtosis": float(sp.kurtosis(valid, bias=False)),
            }
        )
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_skew_kurt({}) -> Item skewness and kurtosis."
