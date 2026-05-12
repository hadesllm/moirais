# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Alert timeline per individual."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def altmrng(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> pd.DataFrame:
    """Alert timeline per individual.

    For each individual, computes the earliest and latest year in which
    any alert was active, plus duration (last - first + 1).

    Parameters
    ----------
    df : DataFrame
        Correctional placement data with alert indicators.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    alert_mh_col, alert_sr_col, alert_sw_col : str
        Alert indicator columns.

    Returns
    -------
    DataFrame
        Columns: id, first_alert_year, last_alert_year, duration.
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    data["_any_alert"] = (data[acols].sum(axis=1) > 0).astype(int)
    active = data[data["_any_alert"] == 1]

    if active.empty:
        return pd.DataFrame(columns=["id", "first_alert_year", "last_alert_year", "duration"])

    result = active.groupby(id_col)[year_col].agg(first_alert_year="min", last_alert_year="max").reset_index()
    result.columns = ["id", "first_alert_year", "last_alert_year"]
    result["duration"] = result["last_alert_year"] - result["first_alert_year"] + 1
    return result


short = altmrng


def cheatsheet() -> str:
    return "altmrng({}) -> Alert timeline per individual."
