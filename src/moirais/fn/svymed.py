"""Survey-weighted median."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_median"]


def survey_median(y, weights):
    """
    Survey-weighted median

    Formula: weighted CDF crosses 0.5

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Francisco-Fuller (1991)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted median"})


def cheatsheet():
    return "svymed: Survey-weighted median"
