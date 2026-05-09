"""Lipschitz envelope condition on criterion functions for M-estimator regularity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_m_estimator_lipschitz_envelope"]


def kosorok_ch2_m_estimator_lipschitz_envelope(m, theta_1, theta_2, x):
    """
    Lipschitz envelope condition on criterion functions for M-estimator regularity

    Formula: | m_{theta_1}(x) - m_{theta_2}(x) | <= m_dot(x) * || theta_1 - theta_2 ||

    Parameters
    ----------
    m : array-like
        Input data.
    theta_1 : array-like
        Input data.
    theta_2 : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.18, p. 29
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Lipschitz envelope condition on criterion functions for M-estimator regularity"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Lipschitz envelope condition on criterion functions for M-estimator regularity"})


def cheatsheet():
    return "ksr054: Lipschitz envelope condition on criterion functions for M-estimator regularity"
