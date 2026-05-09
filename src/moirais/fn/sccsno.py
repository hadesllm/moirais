"""SCCS without event-dependent observation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sccs_no_replacement"]


def sccs_no_replacement(events, periods, person_id):
    """
    SCCS without event-dependent observation

    Formula: per-period rate ratio within person

    Parameters
    ----------
    events : array-like
        Input data.
    periods : array-like
        Input data.
    person_id : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Farrington (1995)
    """
    events = np.atleast_1d(np.asarray(events, dtype=float))
    n = len(events)
    result = float(np.mean(events))
    se = float(np.std(events, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SCCS without event-dependent observation"})


def cheatsheet():
    return "sccsno: SCCS without event-dependent observation"
