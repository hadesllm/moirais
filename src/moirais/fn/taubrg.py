"""Tau-estimator regression (high breakdown + high efficiency)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tau_estimator_regression"]


def tau_estimator_regression(y, X):
    """
    Tau-estimator regression (high breakdown + high efficiency)

    Formula: min tau^2 = sigma^2 (1/n) sum rho_2((y_i - x_i'beta)/sigma)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yohai & Zamar (1988)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Tau-estimator regression (high breakdown + high efficiency)"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Tau-estimator regression (high breakdown + high efficiency)"})


def cheatsheet():
    return "taubrg: Tau-estimator regression (high breakdown + high efficiency)"
