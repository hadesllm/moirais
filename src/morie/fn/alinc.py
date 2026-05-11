# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""New alert incidence rate per year."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def alincd(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> pd.DataFrame:
    """New alert incidence rate per year.

    Counts individuals who transitioned from zero active alerts in
    the prior year to one or more active alerts in the current year.
    First observed year for each individual is excluded (no prior
    reference).

    Parameters
    ----------
    df : DataFrame
        Correctional placement data, multiple rows per individual over years.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    alert_mh_col, alert_sr_col, alert_sw_col : str
        Alert indicator columns.

    Returns
    -------
    DataFrame
        Columns: year, new_cases, n_at_risk, incidence_rate.
    """
    data = df.copy()
    acols = [alert_mh_col, alert_sr_col, alert_sw_col]
    for c in acols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    data["_n_alerts"] = data[acols].sum(axis=1).astype(int)

    # Max alert count per person-year
    yearly = data.groupby([id_col, year_col])["_n_alerts"].max().reset_index().sort_values([id_col, year_col])

    yearly["_prev"] = yearly.groupby(id_col)["_n_alerts"].shift(1)
    transitions = yearly.dropna(subset=["_prev"])
    new_cases = transitions[(transitions["_prev"] == 0) & (transitions["_n_alerts"] > 0)]

    inc = new_cases.groupby(year_col).size().reset_index(name="new_cases")
    at_risk = transitions[transitions["_prev"] == 0].groupby(year_col).size().reset_index(name="n_at_risk")

    result = inc.merge(at_risk, on=year_col, how="outer").fillna(0)
    result.columns = ["year", "new_cases", "n_at_risk"]
    result["new_cases"] = result["new_cases"].astype(int)
    result["n_at_risk"] = result["n_at_risk"].astype(int)
    result["incidence_rate"] = result["new_cases"] / result["n_at_risk"].replace(0, 1)
    return result.sort_values("year").reset_index(drop=True)


short = alincd


def cheatsheet() -> str:
    return "alincd({}) -> New alert incidence rate per year."
