# moirais.fn — function file (hadesllm/moirais)
"""Custody segregation indicator proportion."""

from __future__ import annotations

import pandas as pd
from ._richresult import RichResult


def custody_segregation(
    df: pd.DataFrame,
    *,
    alert_col: str = "alert_mental_health",
    id_col: str = "unique_individual_id",
) -> dict:
    """Proportion of individuals with a segregation-related alert.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    alert_col : str
        Binary alert column (1 = flagged).
    id_col : str
        Unique individual identifier column.

    Returns
    -------
    dict
        Keys: ``n_flagged``, ``n_total``, ``proportion``.
    """
    ind = df.groupby(id_col)[alert_col].max()
    n_flagged = int((ind == 1).sum())
    n_total = len(ind)
    prop = n_flagged / n_total if n_total > 0 else 0.0
    return RichResult(payload={"n_flagged": n_flagged, "n_total": n_total, "proportion": prop})


def cheatsheet() -> str:
    return "custody_segregation({}) -> Custody segregation indicator proportion."
