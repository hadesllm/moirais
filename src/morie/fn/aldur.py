# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Mean alert duration across individuals."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS
from ._richresult import RichResult


def aldurn(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> dict[str, float]:
    """Mean alert duration across individuals.

    For each individual, counts the number of distinct years with at
    least one active alert. Returns mean, median, and SD across all
    individuals who had any alert.

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
    dict
        Keys: mean_years, median_years, sd_years, n_individuals.
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    data["_any_alert"] = (data[acols].sum(axis=1) > 0).astype(int)
    active = data[data["_any_alert"] == 1]

    if active.empty:
        return RichResult(payload={"mean_years": 0.0, "median_years": 0.0, "sd_years": 0.0, "n_individuals": 0})

    years_per_person = active.groupby(id_col)[year_col].nunique()
    return {
        "mean_years": float(years_per_person.mean()),
        "median_years": float(years_per_person.median()),
        "sd_years": float(years_per_person.std()),
        "n_individuals": int(years_per_person.count()),
    }


short = aldurn


def cheatsheet() -> str:
    return "aldurn({}) -> Mean alert duration across individuals."
