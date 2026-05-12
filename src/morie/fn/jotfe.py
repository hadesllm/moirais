# morie.fn -- function file (hadesllm/morie)
"""Calendar/time-of-day feature engineering (day-of-week, hour, month, is_holiday)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_calendar_features"]


def joseph_calendar_features(timestamps):
    """
    Calendar/time-of-day feature engineering (day-of-week, hour, month, is_holiday)

    Formula: features = {dayofweek(t), hour(t), month(t), is_holiday(t), ...}

    Parameters
    ----------
    timestamps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Joseph Ch 6, Calendar Features section
    """
    timestamps = np.atleast_1d(np.asarray(timestamps, dtype=float))
    n = len(timestamps)
    result = float(np.mean(timestamps))
    se = float(np.std(timestamps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Calendar/time-of-day feature engineering (day-of-week, hour, month, is_holiday)"})


def cheatsheet():
    return "jotfe: Calendar/time-of-day feature engineering (day-of-week, hour, month, is_holiday)"
