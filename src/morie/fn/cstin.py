# morie.fn -- function file (hadesllm/morie)
"""Custody incident rate per 1000 person-days."""

from __future__ import annotations

import pandas as pd
from ._richresult import RichResult


def custody_incident_rate(
    df: pd.DataFrame,
    *,
    event_col: str = "D",
    time_col: str = "sentence_days",
) -> dict:
    """Events per 1000 person-days.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    event_col : str
        Binary column indicating event occurrence (1 = event).
    time_col : str
        Column with person-time (sentence days).

    Returns
    -------
    dict
        Keys: ``events``, ``person_days``, ``rate_per_1000``.
    """
    events = int(df[event_col].sum())
    person_days = float(df[time_col].sum())
    rate = (events / person_days * 1000) if person_days > 0 else 0.0
    return RichResult(payload={"events": events, "person_days": person_days, "rate_per_1000": rate})


def cheatsheet() -> str:
    return "custody_incident_rate({}) -> Custody incident rate per 1000 person-days."
