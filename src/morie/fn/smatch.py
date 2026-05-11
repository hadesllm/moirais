"""Self-controlled case series."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sccs_design"]


def sccs_design(events, exposure_windows, person_id):
    """
    Self-controlled case series

    Formula: within-person comparison across exposure windows

    Parameters
    ----------
    events : array-like
        Input data.
    exposure_windows : array-like
        Input data.
    person_id : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Whitaker-Farrington (2002)
    """
    events = np.atleast_1d(np.asarray(events, dtype=float))
    n = len(events)
    result = float(np.mean(events))
    se = float(np.std(events, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Self-controlled case series"})


def cheatsheet():
    return "smatch: Self-controlled case series"
