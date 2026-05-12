# morie.fn -- function file (hadesllm/morie)
"""Custody readmission rate."""

from __future__ import annotations

import pandas as pd
from ._richresult import RichResult


def custody_readmit(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
) -> dict:
    """Readmission rate -- proportion of individuals appearing in 2+ fiscal years.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.

    Returns
    -------
    dict
        Keys: ``n_total``, ``n_readmitted``, ``readmission_rate``.
    """
    years_per = df.groupby(id_col)[year_col].nunique()
    n_total = len(years_per)
    n_readmitted = int((years_per > 1).sum())
    rate = n_readmitted / n_total if n_total > 0 else 0.0
    return RichResult(payload={"n_total": n_total, "n_readmitted": n_readmitted, "readmission_rate": rate})


def cheatsheet() -> str:
    return "custody_readmit({}) -> Custody readmission rate."
