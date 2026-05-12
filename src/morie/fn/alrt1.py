# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Mental health alert prevalence by year."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def alrt_mh(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    alert_col: str = DEFAULT_COLS["alert_mh"],
) -> pd.DataFrame:
    """Mental health alert prevalence by year.

    Counts the number and proportion of unique individuals with
    mental health alert = 1, grouped by fiscal year.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data with alert indicators.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    alert_col : str
        Mental health alert column (binary 0/1 or Yes/No).

    Returns
    -------
    DataFrame
        Columns: year, n_alert, n_total, prevalence.
    """
    data = df.copy()
    if pd.api.types.is_string_dtype(data[alert_col]):
        data[alert_col] = (data[alert_col].str.lower() == "yes").astype(int)

    yearly = (
        data.groupby(year_col)
        .agg(
            n_alert=(alert_col, "sum"),
            n_total=(id_col, "nunique"),
        )
        .reset_index()
    )
    yearly.columns = ["year", "n_alert", "n_total"]
    yearly["n_alert"] = yearly["n_alert"].astype(int)
    yearly["prevalence"] = yearly["n_alert"] / yearly["n_total"]
    return yearly


short = alrt_mh


def cheatsheet() -> str:
    return "alrt_mh({}) -> Mental health alert prevalence by year."
