# morie.fn — function file (hadesllm/morie)
"""Sum score for each respondent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_sum(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> pd.Series:
    """Sum score across items for each respondent.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    items : list[str] or None
        Subset of columns to sum. If None, use all columns.

    Returns
    -------
    Series
        Sum score per respondent.

    References
    ----------
    Streiner, D. L. et al. (2015). Health Measurement Scales (5th ed.).
    Oxford University Press.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])
    subset = data[items] if items is not None else data.select_dtypes(include=[np.number])
    return subset.sum(axis=1)


def cheatsheet() -> str:
    return "score_sum({}) -> Sum score for each respondent."
