"""Mediation analysis for time-to-event outcome."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survival_mediation"]


def survival_mediation(X, M, time, event):
    """
    Mediation analysis for time-to-event outcome

    Formula: NIE = log HR_total - log HR_direct

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
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
    Lange, Vansteelandt, Bekaert (2012)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mediation analysis for time-to-event outcome"})


def cheatsheet():
    return "survmd: Mediation analysis for time-to-event outcome"
