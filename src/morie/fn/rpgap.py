# morie.fn — function file (hadesllm/morie)
"""Mean gap between repeat placements."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_gap(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    date_col: str = "placement_start_date",
) -> pd.DataFrame:
    """Compute mean inter-placement gap in days for repeat offenders.

    Only individuals with two or more placement records are included.
    The gap is the number of days between consecutive placement start
    dates for each individual.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data (one row per placement event).
    id_col : str
        Column with unique individual identifiers.
    date_col : str
        Column with placement event dates.

    Returns
    -------
    DataFrame
        Columns: ``id``, ``n_placements``, ``mean_gap_days``,
        ``median_gap_days``, ``min_gap_days``, ``max_gap_days``.
    """
    tmp = df[[id_col, date_col]].dropna(subset=[date_col]).copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values([id_col, date_col])

    # Keep only individuals with 2+ records
    counts = tmp.groupby(id_col).size()
    repeat_ids = counts[counts >= 2].index
    tmp = tmp[tmp[id_col].isin(repeat_ids)]

    # Compute gaps within each individual
    tmp["gap_days"] = tmp.groupby(id_col)[date_col].diff().dt.days

    stats = (
        tmp.dropna(subset=["gap_days"])
        .groupby(id_col)["gap_days"]
        .agg(
            n_placements="count",
            mean_gap_days="mean",
            median_gap_days="median",
            min_gap_days="min",
            max_gap_days="max",
        )
        .reset_index()
    )
    # n_placements from agg counts gaps; actual placements = gaps + 1
    stats["n_placements"] = stats["n_placements"] + 1
    stats = stats.rename(columns={id_col: "id"})
    return stats.reset_index(drop=True)


rpgap = rplace_gap


def cheatsheet() -> str:
    return "rplace_gap({}) -> Mean gap between repeat placements."
