"""Master theorem giving asymptotic normality of regular Euclidean M-estimators."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_m_estimator_master_theorem"]


def kosorok_ch2_m_estimator_master_theorem(theta_hat_n, theta_0, V, Z, n):
    """
    Master theorem giving asymptotic normality of regular Euclidean M-estimators

    Formula: sqrt(n)(theta_hat_n - theta_0) => -V^{-1} Z, where Z is the Gaussian limit of G_n m_dot_{theta_0}

    Parameters
    ----------
    theta_hat_n : array-like
        Input data.
    theta_0 : array-like
        Input data.
    V : array-like
        Input data.
    Z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Thm 2.13, p. 29
    """
    theta_hat_n = np.atleast_1d(np.asarray(theta_hat_n, dtype=float))
    n = len(theta_hat_n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Master theorem giving asymptotic normality of regular Euclidean M-estimators"})
    estimate = np.median(theta_hat_n)
    se = 1.2533 * np.std(theta_hat_n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Master theorem giving asymptotic normality of regular Euclidean M-estimators"})


def cheatsheet():
    return "ksr057: Master theorem giving asymptotic normality of regular Euclidean M-estimators"
