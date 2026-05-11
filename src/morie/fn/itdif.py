# morie.fn — function file (hadesllm/morie)
"""Item difficulty (classical)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_difficulty(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Classical item difficulty: mean / max for each item.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, difficulty.

    References
    ----------
    Crocker, L. & Algina, J. (2006). Introduction to Classical and Modern
    Test Theory. Cengage Learning.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        mx = X[:, j].max()
        diff = float(X[:, j].mean() / mx) if mx > 0 else np.nan
        rows.append({"item": names[j], "difficulty": diff})
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_difficulty({}) -> Item difficulty (classical)."
