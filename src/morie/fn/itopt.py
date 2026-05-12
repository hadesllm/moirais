# morie.fn -- function file (hadesllm/morie)
"""Item response option frequencies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_option_freq(data: pd.DataFrame | np.ndarray) -> dict[str, pd.DataFrame]:
    """Response option frequency table for each item.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    dict[str, DataFrame]
        Key = item name, value = DataFrame with columns: option, count, pct.

    References
    ----------
    Osterlind, S. J. (2010). Modern Measurement: Theory, Principles, and
    Applications of Mental Appraisal (2nd ed.). Pearson.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])

    result = {}
    for col in data.columns:
        vc = data[col].value_counts(dropna=False).sort_index()
        total = vc.sum()
        df = pd.DataFrame(
            {
                "option": vc.index,
                "count": vc.values,
                "pct": (vc.values / total * 100) if total > 0 else vc.values * 0.0,
            }
        )
        result[col] = df
    return result


def cheatsheet() -> str:
    return "item_option_freq({}) -> Item response option frequencies."
