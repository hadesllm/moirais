# morie.fn -- function file (hadesllm/morie)
"""Temporal crash patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def mto_temporal_pattern(
    datetime_array: np.ndarray | pd.Series | list,
) -> DescriptiveResult:
    """Analyse temporal patterns in crashes (hour, day of week, month).

    Parameters
    ----------
    datetime_array : array-like
        Datetime values of crashes.

    Returns
    -------
    DescriptiveResult
    """
    dt = pd.to_datetime(pd.Series(datetime_array), errors="coerce").dropna()
    if len(dt) == 0:
        raise ValueError("No valid datetimes")
    hour_counts = dt.dt.hour.value_counts().sort_index().to_dict()
    dow_counts = dt.dt.dayofweek.value_counts().sort_index().to_dict()
    month_counts = dt.dt.month.value_counts().sort_index().to_dict()
    peak_hour = max(hour_counts, key=hour_counts.get)
    return DescriptiveResult(
        name="temporal_crash_pattern",
        value=float(peak_hour),
        extra={
            "by_hour": hour_counts,
            "by_dow": dow_counts,
            "by_month": month_counts,
            "peak_hour": int(peak_hour),
            "n": len(dt),
        },
    )


mtotm = mto_temporal_pattern


def cheatsheet() -> str:
    return "mto_temporal_pattern({}) -> Temporal crash patterns."
