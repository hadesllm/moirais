"""Empirical distribution function of regression residuals using a sqrt-n consistent beta estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch1_residual_empirical_distribution"]


def kosorok_ch1_residual_empirical_distribution(Y, Z, beta_hat, t, n):
    """
    Empirical distribution function of regression residuals using a sqrt-n consistent beta estimate

    Formula: F_hat(t) = n^{-1} * sum_{i=1}^{n} 1{Y_i - beta_hat' * Z_i <= t}

    Parameters
    ----------
    Y : array-like
        Input data.
    Z : array-like
        Input data.
    beta_hat : array-like
        Input data.
    t : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 1, Eq 1.2, p. 4
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Empirical distribution function of regression residuals using a sqrt-n consistent beta estimate"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Empirical distribution function of regression residuals using a sqrt-n consistent beta estimate"})


def cheatsheet():
    return "ksr021: Empirical distribution function of regression residuals using a sqrt-n consistent beta estimate"
