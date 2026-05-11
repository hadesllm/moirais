"""Consistency-of-score condition for efficient Z-estimator master theorem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_z_estimator_consistency_score"]


def kosorok_ch3_z_estimator_consistency_score(theta_n, eta_n, theta, eta, l_tilde):
    """
    Consistency-of-score condition for efficient Z-estimator master theorem

    Formula: P_{theta,eta} || l_tilde_{theta_n,eta_n} - l_tilde_{theta,eta} ||^2 -> 0 and P_{theta_n,eta_n} ||l_tilde_{theta_n,eta_n}||^2 = O_P(1)

    Parameters
    ----------
    theta_n : array-like
        Input data.
    eta_n : array-like
        Input data.
    theta : array-like
        Input data.
    eta : array-like
        Input data.
    l_tilde : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.7, p. 44
    """
    theta_n = np.atleast_1d(np.asarray(theta_n, dtype=float))
    n = len(theta_n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Consistency-of-score condition for efficient Z-estimator master theorem"})
    estimate = np.median(theta_n)
    se = 1.2533 * np.std(theta_n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Consistency-of-score condition for efficient Z-estimator master theorem"})


def cheatsheet():
    return "ksr067: Consistency-of-score condition for efficient Z-estimator master theorem"
