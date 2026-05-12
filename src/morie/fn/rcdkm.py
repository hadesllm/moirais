# morie.fn -- function file (hadesllm/morie)
"""Kaplan-Meier survival curve for recidivism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def recidivism_km(
    df: pd.DataFrame,
    *,
    time_col: str = DEFAULT_COLS["sentence"],
    event_col: str = DEFAULT_COLS["treatment"],
) -> dict:
    """Kaplan-Meier survival curve for recidivism.

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
    dict
        times (sorted unique event times), survival (KM estimates),
        ci_lower, ci_upper (Greenwood 95% CI), n_at_risk.
    """
    tmp = df[[time_col, event_col]].dropna()
    times_arr = tmp[time_col].values.astype(float)
    events_arr = tmp[event_col].values.astype(int)

    order = np.argsort(times_arr)
    times_arr = times_arr[order]
    events_arr = events_arr[order]

    unique_times = np.unique(times_arr[events_arr == 1])
    n = len(times_arr)

    km_times = []
    km_surv = []
    km_lo = []
    km_hi = []
    km_nar = []

    surv = 1.0
    var_sum = 0.0

    for t in unique_times:
        at_risk = int(np.sum(times_arr >= t))
        events_at_t = int(np.sum((times_arr == t) & (events_arr == 1)))
        if at_risk == 0:
            continue
        surv *= 1.0 - events_at_t / at_risk
        if at_risk > events_at_t:
            var_sum += events_at_t / (at_risk * (at_risk - events_at_t))
        se = surv * np.sqrt(var_sum)
        km_times.append(float(t))
        km_surv.append(surv)
        km_lo.append(max(0.0, surv - 1.96 * se))
        km_hi.append(min(1.0, surv + 1.96 * se))
        km_nar.append(at_risk)

    return {
        "times": km_times,
        "survival": km_surv,
        "ci_lower": km_lo,
        "ci_upper": km_hi,
        "n_at_risk": km_nar,
    }


rcdkm = recidivism_km


def cheatsheet() -> str:
    return "recidivism_km({}) -> Kaplan-Meier survival curve for recidivism."
