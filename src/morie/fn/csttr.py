# morie.fn -- function file (hadesllm/morie)
"""Custody transfer rate between regions per year."""

from __future__ import annotations

import pandas as pd


def custody_transfer_rate(
    df: pd.DataFrame,
    *,
    region_col: str = "region",
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
) -> pd.DataFrame:
    """Transfer rate -- individuals appearing in 2+ regions within a year.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    region_col : str
        Region column.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, 'n_individuals', 'n_transferred', 'transfer_rate']``.
    """
    rows = []
    for yr, grp in df.groupby(year_col):
        per_ind = grp.groupby(id_col)[region_col].nunique()
        n_total = len(per_ind)
        n_transferred = int((per_ind > 1).sum())
        rate = n_transferred / n_total if n_total > 0 else 0.0
        rows.append({year_col: yr, "n_individuals": n_total, "n_transferred": n_transferred, "transfer_rate": rate})
    return pd.DataFrame(rows).sort_values(year_col).reset_index(drop=True)


def cheatsheet() -> str:
    return "custody_transfer_rate({}) -> Custody transfer rate between regions per year."
