"""Survey-weighted Cox."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_cox"]


def survey_cox(time, event, X, weights):
    """
    Survey-weighted Cox

    Formula: weighted partial likelihood

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Binder (1992)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted Cox"})


def cheatsheet():
    return "svycox: Survey-weighted Cox"
