# morie.fn — function file (hadesllm/morie)
"""Percent of Maximum Possible (PMP) score."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_pmp(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
    min_val: float = 1.0,
    max_val: float = 5.0,
) -> pd.Series:
    """Percent of Maximum Possible score per respondent.

    PMP = (observed - min_possible) / (max_possible - min_possible) * 100

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    items : list[str] or None
        Subset of columns. If None, use all numeric columns.
    min_val : float
        Minimum possible response value (default 1).
    max_val : float
        Maximum possible response value (default 5).

    Returns
    -------
    Series
        PMP score (0-100) per respondent.

    References
    ----------
    Cohen, P. et al. (1999). The problem of units and the circumstance
    for POMP. Multivariate Behavioral Research, 34(3), 315-346.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])
    subset = data[items] if items is not None else data.select_dtypes(include=[np.number])
    k = subset.shape[1]
    min_possible = min_val * k
    max_possible = max_val * k
    denom = max_possible - min_possible
    if abs(denom) < 1e-15:
        return pd.Series(np.full(len(data), np.nan), index=data.index)
    observed = subset.sum(axis=1)
    return (observed - min_possible) / denom * 100


def cheatsheet() -> str:
    return "score_pmp({}) -> Percent of Maximum Possible (PMP) score."
