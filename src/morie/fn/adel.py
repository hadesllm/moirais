# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Alpha if item deleted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn.crba import crba


def adel(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Cronbach's alpha if each item is deleted.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, adel (alpha with that item removed).
    """
    X = np.asarray(data, dtype=np.float64)
    k = X.shape[1]
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]

    rows = []
    for j in range(k):
        a = crba(np.delete(X, j, axis=1)).raw
        rows.append({"item": names[j], "adel": a})
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "adel({}) -> Alpha if item deleted."
