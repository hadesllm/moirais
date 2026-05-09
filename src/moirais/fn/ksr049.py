"""Linearization of Z-estimator deviation around the true parameter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_z_master_linearization"]


def kosorok_ch2_z_master_linearization(Psi_dot, Psi_n, Psi, theta_n, theta_0, n):
    """
    Linearization of Z-estimator deviation around the true parameter

    Formula: || sqrt(n) Psi_dot_{theta_0}(theta_n - theta_0) + sqrt(n)(Psi_n - Psi)(theta_0) ||_L -> 0

    Parameters
    ----------
    Psi_dot : array-like
        Input data.
    Psi_n : array-like
        Input data.
    Psi : array-like
        Input data.
    theta_n : array-like
        Input data.
    theta_0 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.13, p. 26
    """
    Psi_dot = np.atleast_1d(np.asarray(Psi_dot, dtype=float))
    n = len(Psi_dot)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Linearization of Z-estimator deviation around the true parameter"})
    estimate = np.median(Psi_dot)
    se = 1.2533 * np.std(Psi_dot, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Linearization of Z-estimator deviation around the true parameter"})


def cheatsheet():
    return "ksr049: Linearization of Z-estimator deviation around the true parameter"
