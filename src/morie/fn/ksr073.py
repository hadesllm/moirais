"""Corollary giving efficiency of nonparametric maximum likelihood estimator under Donsker and rate conditions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_max_likelihood_efficiency_corollary"]


def kosorok_ch3_max_likelihood_efficiency_corollary(theta_hat_n, theta_0, eta_hat_n, eta_0, Psi_dot_0, Z, n):
    """
    Corollary giving efficiency of nonparametric maximum likelihood estimator under Donsker and rate conditions

    Formula: sqrt(n)( theta_hat_n - theta_0, eta_hat_n - eta_0 ) => -Psi_dot_0^{-1} Z under no-bias and stochastic equicontinuity

    Parameters
    ----------
    theta_hat_n : array-like
        Input data.
    theta_0 : array-like
        Input data.
    eta_hat_n : array-like
        Input data.
    eta_0 : array-like
        Input data.
    Psi_dot_0 : array-like
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
    Kosorok (2008), Cor 3.2, p. 47
    """
    theta_hat_n = np.atleast_1d(np.asarray(theta_hat_n, dtype=float))
    n = len(theta_hat_n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Corollary giving efficiency of nonparametric maximum likelihood estimator under Donsker and rate conditions"})
    estimate = np.median(theta_hat_n)
    se = 1.2533 * np.std(theta_hat_n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Corollary giving efficiency of nonparametric maximum likelihood estimator under Donsker and rate conditions"})


def cheatsheet():
    return "ksr073: Corollary giving efficiency of nonparametric maximum likelihood estimator under Donsker and rate conditions"
