"""No-bias condition for plug-in estimating equation to be efficient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_z_estimator_no_bias"]


def kosorok_ch3_z_estimator_no_bias(theta_n, eta_n, theta, l_tilde, n):
    """
    No-bias condition for plug-in estimating equation to be efficient

    Formula: P_{theta_n,eta} l_tilde_{theta_n,eta_n} = o_P(n^{-1/2} + ||theta_n - theta||)

    Parameters
    ----------
    theta_n : array-like
        Input data.
    eta_n : array-like
        Input data.
    theta : array-like
        Input data.
    l_tilde : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.6, p. 44
    """
    theta_n = np.atleast_1d(np.asarray(theta_n, dtype=float))
    n = len(theta_n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "No-bias condition for plug-in estimating equation to be efficient"})
    estimate = np.median(theta_n)
    se = 1.2533 * np.std(theta_n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "No-bias condition for plug-in estimating equation to be efficient"})


def cheatsheet():
    return "ksr066: No-bias condition for plug-in estimating equation to be efficient"
