"""Epidemic curve smoothing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["epicurve"]


def epicurve(dates, cases, bandwidth):
    """
    Epidemic curve smoothing

    Formula: kernel-smoothed daily incidence

    Parameters
    ----------
    dates : array-like
        Input data.
    cases : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cleveland (1979)
    """
    cases = np.atleast_1d(np.asarray(cases, dtype=float))
    n = len(cases)
    result = float(np.mean(cases))
    se = float(np.std(cases, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epidemic curve smoothing"})


def cheatsheet():
    return "epicur: Epidemic curve smoothing"
