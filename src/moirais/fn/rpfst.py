# moirais.fn — function file (hadesllm/moirais)
"""Time to first placement for each individual."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_first(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    date_col: str = "placement_start_date",
    origin_col: str = "intake_date",
) -> pd.DataFrame:
    """Compute days from origin to first placement per individual.

    For each individual, finds the earliest ``date_col`` value and
    subtracts their ``origin_col`` value.  If no origin column exists,
    the earliest record date across the entire dataset is used.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    date_col : str
        Column with placement event dates.
    origin_col : str
        Column with individual origin / intake date.  If absent from *df*,
        the global minimum of ``date_col`` is used as origin for everyone.

    Returns
    -------
    DataFrame
        Columns: ``id``, ``first_placement_date``, ``origin_date``,
        ``days_to_first``.
    """
    tmp = df[[id_col, date_col]].copy()
    if origin_col in df.columns:
        tmp[origin_col] = pd.to_datetime(df[origin_col])
    tmp[date_col] = pd.to_datetime(tmp[date_col])

    first = tmp.groupby(id_col)[date_col].min().reset_index()
    first.columns = [id_col, "first_placement_date"]

    if origin_col in df.columns:
        origins = tmp.groupby(id_col)[origin_col].min().reset_index()
        origins.columns = [id_col, "origin_date"]
        first = first.merge(origins, on=id_col, how="left")
    else:
        first["origin_date"] = tmp[date_col].min()

    first["days_to_first"] = (first["first_placement_date"] - first["origin_date"]).dt.days

    first = first.rename(columns={id_col: "id"})
    return first[first["days_to_first"].notna()].reset_index(drop=True)


rpfst = rplace_first


def cheatsheet() -> str:
    return "rplace_first({}) -> Time to first placement for each individual."
