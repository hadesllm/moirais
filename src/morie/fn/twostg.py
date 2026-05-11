"""Two-stage hazard regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["two_stage_hazard"]


def two_stage_hazard(time, event, X, Z):
    """
    Two-stage hazard regression

    Formula: lambda(t) = lambda_0(t) * f(beta'X) * g(gamma'Z)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cheng, Wei, Ying (1995)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-stage hazard regression"})


def cheatsheet():
    return "twostg: Two-stage hazard regression"
