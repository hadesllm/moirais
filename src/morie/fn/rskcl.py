# morie.fn — function file (hadesllm/morie)
"""Classify individuals into risk levels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def risk_classify(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Classify into Low/Medium/High risk based on score thresholds.

    Parameters
    ----------
    df : DataFrame
        Dataset with risk score column.
    score_col : str
        Column with continuous risk score.
    thresholds : list of float, optional
        Two cutpoints [low_high_boundary, high_boundary].
        Defaults to terciles of the score distribution.

    Returns
    -------
    DataFrame
        Original DataFrame with added ``risk_level`` column.
    """
    result = df.copy()
    scores = result[score_col]
    if thresholds is None:
        thresholds = [float(np.nanpercentile(scores, 33)), float(np.nanpercentile(scores, 67))]
    conditions = [
        scores <= thresholds[0],
        scores <= thresholds[1],
    ]
    choices = ["Low", "Medium"]
    result["risk_level"] = np.select(conditions, choices, default="High")
    return result


rskcl = risk_classify


def cheatsheet() -> str:
    return "risk_classify({}) -> Classify individuals into risk levels."
