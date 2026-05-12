# morie.fn -- function file (hadesllm/morie)
"""Outcome rate by risk score decile."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def risk_decile(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    outcome_col: str = DEFAULT_COLS["treatment"],
) -> pd.DataFrame:
    """Outcome rate by risk score decile.

    Parameters
    ----------
    df : DataFrame
        Dataset with score and binary outcome columns.
    score_col : str
        Column with continuous risk score.
    outcome_col : str
        Column with binary outcome (1 = event).

    Returns
    -------
    DataFrame
        Columns: decile, mean_score, outcome_rate, n.
    """
    tmp = df[[score_col, outcome_col]].dropna()
    tmp = tmp.assign(_decile=pd.qcut(tmp[score_col], q=10, labels=False, duplicates="drop"))
    grouped = (
        tmp.groupby("_decile")
        .agg(
            mean_score=(score_col, "mean"),
            outcome_rate=(outcome_col, "mean"),
            n=(outcome_col, "count"),
        )
        .reset_index()
    )
    grouped.columns = ["decile", "mean_score", "outcome_rate", "n"]
    return grouped


rskdc = risk_decile


def cheatsheet() -> str:
    return "risk_decile({}) -> Outcome rate by risk score decile."
