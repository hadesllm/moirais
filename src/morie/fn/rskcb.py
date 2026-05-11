# morie.fn — function file (hadesllm/morie)
"""Risk score calibration by decile."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def risk_calibration(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    outcome_col: str = DEFAULT_COLS["treatment"],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calibration table: observed vs predicted outcome rate by score decile.

    Parameters
    ----------
    df : DataFrame
        Dataset with score and binary outcome columns.
    score_col : str
        Column with predicted risk score.
    outcome_col : str
        Column with binary outcome (1 = event).
    n_bins : int
        Number of calibration bins (default 10 = deciles).

    Returns
    -------
    DataFrame
        Columns: bin, mean_predicted, observed_rate, n.
    """
    tmp = df[[score_col, outcome_col]].dropna()
    tmp = tmp.assign(_bin=pd.qcut(tmp[score_col], q=n_bins, labels=False, duplicates="drop"))
    grouped = (
        tmp.groupby("_bin")
        .agg(
            mean_predicted=(score_col, "mean"),
            observed_rate=(outcome_col, "mean"),
            n=(outcome_col, "count"),
        )
        .reset_index()
    )
    grouped.columns = ["bin", "mean_predicted", "observed_rate", "n"]
    return grouped


rskcb = risk_calibration


def cheatsheet() -> str:
    return "risk_calibration({}) -> Risk score calibration by decile."
