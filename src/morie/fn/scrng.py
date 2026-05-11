# morie.fn — function file (hadesllm/morie)
"""Score range check — validate responses within bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_range_check(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
    min_val: float = 1.0,
    max_val: float = 5.0,
) -> dict:
    """Check all responses fall within valid range.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    items : list[str] or None
        Subset of columns. If None, use all numeric columns.
    min_val : float
        Minimum valid response (default 1).
    max_val : float
        Maximum valid response (default 5).

    Returns
    -------
    dict
        Keys: 'n_total', 'n_invalid', 'pct_invalid', 'flagged_rows',
        'flagged_items'.

    References
    ----------
    Tabachnick, B. G. & Fidell, L. S. (2013). Using Multivariate Statistics
    (6th ed.). Pearson.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])
    subset = data[items] if items is not None else data.select_dtypes(include=[np.number])

    n_total = int(subset.size)
    out_of_range = (subset < min_val) | (subset > max_val)
    n_invalid = int(out_of_range.sum().sum())
    pct_invalid = float(n_invalid / n_total * 100) if n_total > 0 else 0.0

    flagged_rows = sorted(int(i) for i in out_of_range.any(axis=1)[out_of_range.any(axis=1)].index)
    flagged_items = [col for col in subset.columns if out_of_range[col].any()]

    return {
        "n_total": n_total,
        "n_invalid": n_invalid,
        "pct_invalid": pct_invalid,
        "flagged_rows": flagged_rows,
        "flagged_items": flagged_items,
    }


def cheatsheet() -> str:
    return "score_range_check({}) -> Score range check — validate responses within bounds."
