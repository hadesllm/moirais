# moirais.fn — function file (hadesllm/moirais)
"""Custody lockdown frequency per year."""

from __future__ import annotations

import pandas as pd


def custody_lockdown_freq(
    df: pd.DataFrame,
    *,
    event_col: str = "D",
    year_col: str = "end_fiscal_year",
) -> pd.DataFrame:
    """Event frequency (lockdown proxy) per year.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    event_col : str
        Binary event column (1 = event).
    year_col : str
        Fiscal year column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, 'n', 'n_events', 'rate']``.
    """
    grp = df.groupby(year_col)[event_col].agg(n="count", n_events="sum").reset_index()
    grp["rate"] = grp["n_events"] / grp["n"]
    return grp.sort_values(year_col).reset_index(drop=True)


def cheatsheet() -> str:
    return "custody_lockdown_freq({}) -> Custody lockdown frequency per year."
