# morie.fn -- function file (hadesllm/morie)
"""Mean risk score profile by demographic subgroups."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def risk_profile(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Mean risk score by demographic subgroups.

    Parameters
    ----------
    df : DataFrame
        Dataset with score column.
    score_col : str
        Column with continuous risk score.
    group_cols : list of str, optional
        Columns to group by. Defaults to ``["region", "age_group", "gender"]``
        (only those present in df).

    Returns
    -------
    DataFrame
        Group columns plus mean_score, sd_score, n.
    """
    if group_cols is None:
        candidates = [DEFAULT_COLS["region"], DEFAULT_COLS["age"], DEFAULT_COLS["gender"]]
        group_cols = [c for c in candidates if c in df.columns]
    if not group_cols:
        group_cols = [df.columns[0]]

    tmp = df[group_cols + [score_col]].dropna()
    grouped = (
        tmp.groupby(group_cols)[score_col]
        .agg(
            mean_score="mean",
            sd_score="std",
            n="count",
        )
        .reset_index()
    )
    return grouped


rskpr = risk_profile


def cheatsheet() -> str:
    return "risk_profile({}) -> Mean risk score profile by demographic subgroups."
