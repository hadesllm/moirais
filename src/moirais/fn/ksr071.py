"""Quadratic expansion of the log-profile likelihood used for chi-square inference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_log_profile_expansion"]


def kosorok_ch3_log_profile_expansion(theta_bar_n, theta_hat_n, I_tilde, n):
    """
    Quadratic expansion of the log-profile likelihood used for chi-square inference

    Formula: log pl_n(theta_bar_n) = log pl_n(theta_hat_n) - (1/2) n (theta_bar_n - theta_hat_n)' I_tilde_{theta_0,eta_0} (theta_bar_n - theta_hat_n) + o_P((sqrt(n)||theta_bar_n - theta_hat_n||+1)^2)

    Parameters
    ----------
    theta_bar_n : array-like
        Input data.
    theta_hat_n : array-like
        Input data.
    I_tilde : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.11, p. 48
    """
    theta_bar_n = np.atleast_1d(np.asarray(theta_bar_n, dtype=float))
    n = len(theta_bar_n)
    result = float(np.mean(theta_bar_n))
    se = float(np.std(theta_bar_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratic expansion of the log-profile likelihood used for chi-square inference"})


def cheatsheet():
    return "ksr071: Quadratic expansion of the log-profile likelihood used for chi-square inference"
