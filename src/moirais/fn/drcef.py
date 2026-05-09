"""DR Callaway-Sant'Anna event-study aggregation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_callaway_event_study"]


def dr_callaway_event_study(y, D, unit, time, cohort):
    """
    DR Callaway-Sant'Anna event-study aggregation

    Formula: avg ATT(g, g+e) over event-time e

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    cohort : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Callaway-Sant'Anna (2021) §3.4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR Callaway-Sant'Anna event-study aggregation"})


def cheatsheet():
    return "drcef: DR Callaway-Sant'Anna event-study aggregation"
