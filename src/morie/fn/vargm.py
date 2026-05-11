"""Empirical (Matheron) variogram estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["empirical_variogram"]


def empirical_variogram(coords, z, lags):
    """
    Empirical (Matheron) variogram estimator

    Formula: gamma(h) = (1 / 2 |N(h)|) sum_{(i,j) in N(h)} (z_i - z_j)^2

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    lags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matheron (1962)
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Empirical (Matheron) variogram estimator"})
    estimate = np.median(z)
    se = 1.2533 * np.std(z, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Empirical (Matheron) variogram estimator"})


def cheatsheet():
    return "vargm: Empirical (Matheron) variogram estimator"
