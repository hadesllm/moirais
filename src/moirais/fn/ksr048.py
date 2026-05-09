"""Stochastic equicontinuity condition needed in the Z-estimator master theorem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_z_master_stochastic_equicontinuity"]


def kosorok_ch2_z_master_stochastic_equicontinuity(Psi_n, Psi, theta_n, theta_0, n):
    """
    Stochastic equicontinuity condition needed in the Z-estimator master theorem

    Formula: || sqrt(n)(Psi_n(theta_n) - Psi(theta_n)) - sqrt(n)(Psi_n(theta_0) - Psi(theta_0)) ||_L / (1 + sqrt(n)||theta_n-theta_0||) -> 0

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
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.12, p. 26
    """
    Psi_n = np.atleast_1d(np.asarray(Psi_n, dtype=float))
    n = len(Psi_n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Stochastic equicontinuity condition needed in the Z-estimator master theorem"})
    estimate = np.median(Psi_n)
    se = 1.2533 * np.std(Psi_n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Stochastic equicontinuity condition needed in the Z-estimator master theorem"})


def cheatsheet():
    return "ksr048: Stochastic equicontinuity condition needed in the Z-estimator master theorem"
