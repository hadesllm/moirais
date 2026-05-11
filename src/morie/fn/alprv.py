# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Alert prevalence by grouping variable."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def alprev(
    df: pd.DataFrame,
    *,
    group_col: str = DEFAULT_COLS["region"],
    id_col: str = DEFAULT_COLS["id"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> pd.DataFrame:
    """Alert prevalence by grouping variable.

    Cross-tabulates alert prevalence (any alert active) by a user-specified
    grouping variable (region, age group, gender, etc.).

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    group_col : str
        Column to group by (e.g. region, age_group, gender).
    id_col : str
        Unique individual identifier column.
    alert_mh_col, alert_sr_col, alert_sw_col : str
        Alert indicator columns.

    Returns
    -------
    DataFrame
        Columns: group, n_any_alert, n_total, prevalence.
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    data["_any_alert"] = (data[acols].sum(axis=1) > 0).astype(int)

    grouped = (
        data.groupby(group_col)
        .agg(
            n_any_alert=("_any_alert", "sum"),
            n_total=(id_col, "count"),
        )
        .reset_index()
    )
    grouped.columns = ["group", "n_any_alert", "n_total"]
    grouped["n_any_alert"] = grouped["n_any_alert"].astype(int)
    grouped["prevalence"] = grouped["n_any_alert"] / grouped["n_total"]
    return grouped


short = alprev


def cheatsheet() -> str:
    return "alprev({}) -> Alert prevalence by grouping variable."
