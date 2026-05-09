# moirais.fn — function file (hadesllm/moirais)
"""Mean score for each respondent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_mean(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> pd.Series:
    """Mean score across items for each respondent (handles missing).

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    items : list[str] or None
        Subset of columns. If None, use all numeric columns.

    Returns
    -------
    Series
        Mean score per respondent.

    References
    ----------
    Schafer, J. L. & Graham, J. W. (2002). Missing data: Our view of the
    state of the art. Psychological Methods, 7(2), 147-177.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])
    subset = data[items] if items is not None else data.select_dtypes(include=[np.number])
    return subset.mean(axis=1)


def cheatsheet() -> str:
    return "score_mean({}) -> Mean score for each respondent."
