# morie.fn -- function file (hadesllm/morie)
"""Base rate of outcome overall and by group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def risk_base_rate(
    df: pd.DataFrame,
    *,
    outcome_col: str = DEFAULT_COLS["treatment"],
    group_col: str | None = None,
) -> dict | pd.DataFrame:
    """Base rate of binary outcome, overall and optionally by group.

    Parameters
    ----------
    df : DataFrame
        Dataset with outcome column.
    outcome_col : str
        Column with binary outcome (1 = event).
    group_col : str, optional
        If provided, returns base rate per group as DataFrame.

    Returns
    -------
    dict or DataFrame
        If group_col is None: dict with base_rate, n_events, n_total.
        If group_col given: DataFrame with group, base_rate, n_events, n_total.
    """
    if group_col is None:
        tmp = df[outcome_col].dropna()
        n_total = len(tmp)
        n_events = int((tmp > 0).sum())
        return {
            "base_rate": n_events / n_total if n_total > 0 else np.nan,
            "n_events": n_events,
            "n_total": n_total,
        }

    tmp = df[[outcome_col, group_col]].dropna()
    tmp = tmp.assign(_event=(tmp[outcome_col] > 0).astype(int))
    grouped = (
        tmp.groupby(group_col)
        .agg(
            n_total=("_event", "count"),
            n_events=("_event", "sum"),
        )
        .reset_index()
    )
    grouped.columns = ["group", "n_total", "n_events"]
    grouped["base_rate"] = grouped["n_events"] / grouped["n_total"]
    return grouped


rskbr = risk_base_rate


def cheatsheet() -> str:
    return "risk_base_rate({}) -> Base rate of outcome overall and by group."
