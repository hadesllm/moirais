# morie.fn — function file (hadesllm/morie)
"""Lexis diagram data preparation."""

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def lexis_diagram_data(
    birth_dates: list | np.ndarray,
    event_dates: list | np.ndarray,
) -> DescriptiveResult:
    """Prepare data for a Lexis diagram (birth date x event date).

    Parameters
    ----------
    birth_dates : array-like
        Birth dates as numeric years (e.g. 1990.5).
    event_dates : array-like
        Event dates as numeric years.

    Returns
    -------
    DescriptiveResult
    """
    bd = np.asarray(birth_dates, dtype=float)
    ed = np.asarray(event_dates, dtype=float)
    if len(bd) != len(ed):
        raise ValueError("birth_dates and event_dates must match")

    age_at_event = ed - bd

    df = pd.DataFrame(
        {
            "birth_date": bd,
            "event_date": ed,
            "age_at_event": age_at_event,
        }
    )

    return DescriptiveResult(
        name="lexis_diagram_data",
        value=df,
        extra={
            "n": len(bd),
            "mean_age_at_event": float(np.mean(age_at_event)),
            "year_range": [float(np.min(ed)), float(np.max(ed))],
        },
    )


lexis = lexis_diagram_data


def cheatsheet() -> str:
    return "lexis_diagram_data({}) -> Lexis diagram data preparation."
