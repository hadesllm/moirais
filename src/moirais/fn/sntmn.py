"""Mandatory minimum sentence analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import DescriptiveResult


def sentence_mandatory_min(
    df: pd.DataFrame,
    *,
    offense_col: str = "offense",
    sentence_col: str = "sentence_days",
    min_col: str = "mandatory_min_days",
) -> DescriptiveResult:
    """Mandatory minimum sentence analysis.

    Parameters
    ----------
    df : DataFrame
    offense_col : str
    sentence_col : str
    min_col : str
        Mandatory minimum days column.

    Returns
    -------
    DescriptiveResult
        extra: pct_at_minimum, mean_above_minimum.
    """
    diff = df[sentence_col] - df[min_col]
    at_min = float(np.mean(diff <= 0))
    above = diff[diff > 0]
    mean_above = float(above.mean()) if len(above) > 0 else 0.0
    return DescriptiveResult(
        name="sentence_mandatory_min",
        value=at_min,
        extra={"pct_at_minimum": at_min, "mean_above_minimum": mean_above, "n": len(df)},
    )


sntmn = sentence_mandatory_min


def cheatsheet() -> str:
    return "sentence_mandatory_min({}) -> Mandatory minimum sentence analysis."
