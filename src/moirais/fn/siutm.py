"""SIU case processing time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import DescriptiveResult


def siu_timeline(
    start_dates: list | np.ndarray | pd.Series,
    end_dates: list | np.ndarray | pd.Series,
) -> DescriptiveResult:
    """Analyse SIU case processing time (days from incident to closure).

    Parameters
    ----------
    start_dates : array-like
        Investigation start dates.
    end_dates : array-like
        Investigation closure dates.

    Returns
    -------
    DescriptiveResult
    """
    starts = pd.to_datetime(pd.Series(start_dates), errors="coerce")
    ends = pd.to_datetime(pd.Series(end_dates), errors="coerce")
    valid = starts.notna() & ends.notna()
    if valid.sum() == 0:
        raise ValueError("No valid date pairs")
    days = (ends[valid] - starts[valid]).dt.days.astype(float)
    days = days[days >= 0]
    if len(days) == 0:
        raise ValueError("No valid positive durations")
    return DescriptiveResult(
        name="siu_processing_time",
        value=float(np.median(days)),
        extra={
            "mean_days": float(np.mean(days)),
            "median_days": float(np.median(days)),
            "std_days": float(np.std(days, ddof=1)) if len(days) > 1 else 0.0,
            "min_days": float(np.min(days)),
            "max_days": float(np.max(days)),
            "n": len(days),
        },
    )


siutm = siu_timeline


def cheatsheet() -> str:
    return "siu_timeline({}) -> SIU case processing time."
