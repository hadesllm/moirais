# morie.fn — function file (hadesllm/morie)
"""Hazard rate at each time point for recidivism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def recidivism_hazard(
    df: pd.DataFrame,
    *,
    time_col: str = DEFAULT_COLS["sentence"],
    event_col: str = DEFAULT_COLS["treatment"],
) -> pd.DataFrame:
    """Hazard rate at each unique event time.

    Computes the discrete hazard h(t) = d(t) / n(t) where d(t) is the
    number of events and n(t) is the number at risk at time t.

    Parameters
    ----------
    df : DataFrame
        Dataset with time and event columns.
    time_col : str
        Column with time-to-event.
    event_col : str
        Column with event indicator (1 = event).

    Returns
    -------
    DataFrame
        Columns: time, n_at_risk, n_events, hazard.
    """
    tmp = df[[time_col, event_col]].dropna()
    times_arr = tmp[time_col].values.astype(float)
    events_arr = tmp[event_col].values.astype(int)

    unique_times = np.sort(np.unique(times_arr[events_arr == 1]))
    rows = []
    for t in unique_times:
        at_risk = int(np.sum(times_arr >= t))
        n_events = int(np.sum((times_arr == t) & (events_arr == 1)))
        hazard = n_events / at_risk if at_risk > 0 else 0.0
        rows.append({"time": t, "n_at_risk": at_risk, "n_events": n_events, "hazard": hazard})
    return pd.DataFrame(rows)


rcdhz = recidivism_hazard


def cheatsheet() -> str:
    return "recidivism_hazard({}) -> Hazard rate at each time point for recidivism."
