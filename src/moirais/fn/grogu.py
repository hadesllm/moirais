# moirais.fn — function file (hadesllm/moirais)
"""Grouped summary statistics. 'The Child' -- Grogu"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def grouped_stats(
    data: pd.DataFrame,
    *,
    by: str = "group",
    cols: list[str] | str | None = None,
) -> DescriptiveResult:
    """Compute mean, std, n for numeric columns by group."""
    _validate_df(data, by)
    if cols is None:
        cols = data.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(cols, str):
        cols = [cols]
    result = data.groupby(by)[cols].agg(["mean", "std", "count"]).reset_index()
    return DescriptiveResult(
        name=f"Grouped by {by}",
        value=result,
        extra={"n_groups": data[by].nunique(), "columns": cols},
    )


grogu = grouped_stats


def cheatsheet() -> str:
    return "grouped_stats({}) -> Grouped summary statistics. 'The Child' -- Grogu"
