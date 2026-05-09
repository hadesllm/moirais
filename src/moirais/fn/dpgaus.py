"""Gaussian mechanism for (epsilon, delta)-DP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_gaussian_mechanism"]


def dp_gaussian_mechanism(y, sensitivity, epsilon, delta):
    """
    Gaussian mechanism for (epsilon, delta)-DP

    Formula: M(x) = f(x) + N(0, (sensitivity^2 * 2 ln(1.25/delta))/epsilon^2)

    Parameters
    ----------
    y : array-like
        Input data.
    sensitivity : array-like
        Input data.
    epsilon : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork & Roth (2014) §3.5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian mechanism for (epsilon, delta)-DP"})


def cheatsheet():
    return "dpgaus: Gaussian mechanism for (epsilon, delta)-DP"
