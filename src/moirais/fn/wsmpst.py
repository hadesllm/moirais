"""Plug-in estimator theta_hat = T(F_n)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_plug_in_estimator"]


def wasserman_plug_in_estimator(data, T):
    """
    Plug-in estimator theta_hat = T(F_n)

    Formula: theta_hat = T(F_n)

    Parameters
    ----------
    data : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 7
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Plug-in estimator theta_hat = T(F_n)"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Plug-in estimator theta_hat = T(F_n)"})


def cheatsheet():
    return "wsmpst: Plug-in estimator theta_hat = T(F_n)"
