# moirais.fn — function file (hadesllm/moirais)
"""Time-to-event summary for recidivism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def recidivism_time(
    df: pd.DataFrame,
    *,
    time_col: str = DEFAULT_COLS["sentence"],
    event_col: str = DEFAULT_COLS["treatment"],
) -> dict:
    """Time-to-event summary statistics for recidivism.

    Parameters
    ----------
    df : DataFrame
        Dataset with time and event columns.
    time_col : str
        Column with time-to-event (e.g. sentence days).
    event_col : str
        Column with event indicator (1 = event, 0 = censored).

    Returns
    -------
    dict
        mean, median, sd, q25, q75, n_events, n_censored, n_total.
    """
    tmp = df[[time_col, event_col]].dropna()
    times = tmp[time_col].values
    events = tmp[event_col].values
    n_total = len(times)
    n_events = int(events.sum())
    n_censored = n_total - n_events

    event_times = times[events == 1] if n_events > 0 else times

    return {
        "mean": float(np.mean(event_times)) if len(event_times) > 0 else np.nan,
        "median": float(np.median(event_times)) if len(event_times) > 0 else np.nan,
        "sd": float(np.std(event_times, ddof=1)) if len(event_times) > 1 else np.nan,
        "q25": float(np.percentile(event_times, 25)) if len(event_times) > 0 else np.nan,
        "q75": float(np.percentile(event_times, 75)) if len(event_times) > 0 else np.nan,
        "n_events": n_events,
        "n_censored": n_censored,
        "n_total": n_total,
    }


rcdtm = recidivism_time


def cheatsheet() -> str:
    return "recidivism_time({}) -> Time-to-event summary for recidivism."
