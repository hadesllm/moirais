# moirais.fn — function file (hadesllm/moirais)
"""Custody time served distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def custody_time_served(
    df: pd.DataFrame,
    *,
    sent_col: str = "sentence_days",
    id_col: str = "unique_individual_id",
) -> dict:
    """Distribution of total time served per individual.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    sent_col : str
        Column with sentence length in days.
    id_col : str
        Unique individual identifier column.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``std``, ``min``, ``max``, ``q25``, ``q75``, ``n``.
    """
    totals = df.groupby(id_col)[sent_col].sum()
    return {
        "mean": float(totals.mean()),
        "median": float(totals.median()),
        "std": float(totals.std()),
        "min": float(totals.min()),
        "max": float(totals.max()),
        "q25": float(np.percentile(totals, 25)),
        "q75": float(np.percentile(totals, 75)),
        "n": len(totals),
    }


def cheatsheet() -> str:
    return "custody_time_served({}) -> Custody time served distribution."
