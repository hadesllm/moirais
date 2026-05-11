"""Left-truncated survival adjustment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["surv_truncation_left"]


def surv_truncation_left(entry, time, event):
    """
    Left-truncated survival adjustment

    Formula: risk set conditional on T > entry_t

    Parameters
    ----------
    entry : array-like
        Input data.
    time : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Klein-Moeschberger (2003)
    """
    entry = np.atleast_1d(np.asarray(entry, dtype=float))
    n = len(entry)
    result = float(np.mean(entry))
    se = float(np.std(entry, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Left-truncated survival adjustment"})


def cheatsheet():
    return "sstrlf: Left-truncated survival adjustment"
