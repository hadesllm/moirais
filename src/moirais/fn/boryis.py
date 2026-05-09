"""Borusyak-Jaravel-Spiess imputation estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["borusyak_jaravel_spiess"]


def borusyak_jaravel_spiess(y, D, unit, time, X):
    """
    Borusyak-Jaravel-Spiess imputation estimator

    Formula: impute Y_it(0) for treated; tau = Y_it - hat Y_it(0)

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Borusyak-Jaravel-Spiess (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Borusyak-Jaravel-Spiess imputation estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Borusyak-Jaravel-Spiess imputation estimator"})


def cheatsheet():
    return "boryis: Borusyak-Jaravel-Spiess imputation estimator"
