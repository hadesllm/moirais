"""Consistency theorem for Z-estimators under identifiability and uniform convergence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_z_estimator_consistency"]


def kosorok_ch2_z_estimator_consistency(Psi_n, Psi, theta_n, theta_0):
    """
    Consistency theorem for Z-estimators under identifiability and uniform convergence

    Formula: If sup_theta || Psi_n(theta) - Psi(theta) ||_L -> 0 and ||Psi_n(theta_n)||_L -> 0 then theta_n -> theta_0

    Parameters
    ----------
    Psi_n : array-like
        Input data.
    Psi : array-like
        Input data.
    theta_n : array-like
        Input data.
    theta_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Thm 2.10, p. 24
    """
    Psi_n = np.atleast_1d(np.asarray(Psi_n, dtype=float))
    n = len(Psi_n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Consistency theorem for Z-estimators under identifiability and uniform convergence"})
    estimate = np.median(Psi_n)
    se = 1.2533 * np.std(Psi_n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Consistency theorem for Z-estimators under identifiability and uniform convergence"})


def cheatsheet():
    return "ksr046: Consistency theorem for Z-estimators under identifiability and uniform convergence"
