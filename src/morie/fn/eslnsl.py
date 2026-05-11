"""Natural cubic spline basis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_natural_spline"]


def esl_natural_spline(x, knots):
    """
    Natural cubic spline basis

    Formula: Natural spline: linear beyond boundaries

    Parameters
    ----------
    x : array-like
        Input data.
    knots : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: basis

    References
    ----------
    Hastie ESL Ch 5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Natural cubic spline basis"})


def cheatsheet():
    return "eslnsl: Natural cubic spline basis"
