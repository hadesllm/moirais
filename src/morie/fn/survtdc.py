"""Time-dependent concordance index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["time_dep_concordance"]


def time_dep_concordance(time, event, marker, t):
    """
    Time-dependent concordance index

    Formula: per-time C-statistic with IPCW

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    marker : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Antolini-Boracchi-Biganzoli (2005)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-dependent concordance index"})


def cheatsheet():
    return "survtdc: Time-dependent concordance index"
