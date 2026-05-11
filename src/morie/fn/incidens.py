"""Incidence rate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["incidence_rate"]


def incidence_rate(cases, person_time):
    """
    Incidence rate

    Formula: IR = new cases / person-time

    Parameters
    ----------
    cases : array-like
        Input data.
    person_time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rothman (2012)
    """
    cases = np.atleast_1d(np.asarray(cases, dtype=float))
    n = len(cases)
    result = float(np.mean(cases))
    se = float(np.std(cases, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Incidence rate"})


def cheatsheet():
    return "incidens: Incidence rate"
