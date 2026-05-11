# morie.fn — function file (hadesllm/morie)
"""Total custody days per individual."""

from __future__ import annotations

import pandas as pd


def custody_days(
    df: pd.DataFrame,
    *,
    sent_col: str = "sentence_days",
    id_col: str = "unique_individual_id",
) -> pd.DataFrame:
    """Total custody days per individual.

    Parameters
    ----------
    df : DataFrame
        Correctional records with sentence length and individual IDs.
    sent_col : str
        Column containing sentence length in days.
    id_col : str
        Column containing unique individual identifiers.

    Returns
    -------
    DataFrame
        Columns: ``[id_col, 'total_days', 'n_records']``.
    """
    out = df.groupby(id_col)[sent_col].agg(total_days="sum", n_records="count").reset_index()
    return out


def cheatsheet() -> str:
    return "custody_days({}) -> Total custody days per individual."
