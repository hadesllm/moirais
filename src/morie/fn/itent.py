# morie.fn — function file (hadesllm/morie)
"""Item response entropy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_entropy(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Shannon entropy of response distribution per item.

    Higher entropy indicates more uniform response spread.
    Maximum entropy = log2(number of unique options).

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, entropy, max_entropy, relative_entropy.

    References
    ----------
    Shannon, C. E. (1948). A mathematical theory of communication.
    Bell System Technical Journal, 27(3), 379-423.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            rows.append({"item": names[j], "entropy": np.nan, "max_entropy": np.nan, "relative_entropy": np.nan})
            continue
        _, counts = np.unique(valid, return_counts=True)
        p = counts / counts.sum()
        h = float(-np.sum(p * np.log2(p + 1e-15)))
        h_max = float(np.log2(len(counts))) if len(counts) > 1 else 0.0
        rel_h = float(h / h_max) if h_max > 0 else 0.0
        rows.append(
            {
                "item": names[j],
                "entropy": h,
                "max_entropy": h_max,
                "relative_entropy": rel_h,
            }
        )
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_entropy({}) -> Item response entropy."
