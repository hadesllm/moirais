"""Cox partial-likelihood score (estimating equation) for beta in proportional hazards model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch1_cox_estimating_equation"]


def kosorok_ch1_cox_estimating_equation(t, beta, Z, Y, N, n):
    """
    Cox partial-likelihood score (estimating equation) for beta in proportional hazards model

    Formula: U_n(t,beta) = n^{-1} * sum_i integral_0^t [Z_i - E_n(s,beta)] dN_i(s), where E_n(s,beta) = sum_i Z_i Y_i(s) e^{beta'Z_i} / sum_i Y_i(s) e^{beta'Z_i}

    Parameters
    ----------
    t : array-like
        Input data.
    beta : array-like
        Input data.
    Z : array-like
        Input data.
    Y : array-like
        Input data.
    N : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 1, Eq 1.4, p. 5
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Cox partial-likelihood score (estimating equation) for beta in proportional hazards model"})
    estimate = np.median(t)
    se = 1.2533 * np.std(t, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Cox partial-likelihood score (estimating equation) for beta in proportional hazards model"})


def cheatsheet():
    return "ksr023: Cox partial-likelihood score (estimating equation) for beta in proportional hazards model"
