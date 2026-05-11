# morie.fn — function file (hadesllm/morie)
"""Inspection fail rate — proportion below threshold."""

from __future__ import annotations

import pandas as pd
from ._richresult import RichResult


def inspection_fail_rate(
    df: pd.DataFrame,
    *,
    score_col: str = "Y",
    threshold: float = 0.0,
) -> dict:
    """Proportion of records where score falls below threshold.

    Parameters
    ----------
    df : DataFrame
        Records with a numeric score.
    score_col : str
        Numeric score column.
    threshold : float
        Score threshold (below = fail).

    Returns
    -------
    dict
        Keys: ``n``, ``n_fail``, ``fail_rate``, ``threshold``.
    """
    n = len(df)
    n_fail = int((df[score_col] < threshold).sum())
    rate = n_fail / n if n > 0 else 0.0
    return RichResult(payload={"n": n, "n_fail": n_fail, "fail_rate": rate, "threshold": threshold})


def cheatsheet() -> str:
    return "inspection_fail_rate({}) -> Inspection fail rate — proportion below threshold."
