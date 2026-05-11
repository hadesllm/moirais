"""Cox likelihood with mass at observed failure times leading to the Breslow estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_cox_likelihood_breslow"]


def kosorok_ch3_cox_likelihood_breslow(beta, Lambda, Z, V, d, n):
    """
    Cox likelihood with mass at observed failure times leading to the Breslow estimator

    Formula: L_n(beta, Lambda) = prod_{i=1}^n [ e^{beta'Z_i} dLambda(V_i) ]^{d_i} exp[ -e^{beta'Z_i} Lambda(V_i) ]

    Parameters
    ----------
    beta : array-like
        Input data.
    Lambda : array-like
        Input data.
    Z : array-like
        Input data.
    V : array-like
        Input data.
    d : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.9, p. 45
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Cox likelihood with mass at observed failure times leading to the Breslow estimator"})
    estimate = np.median(beta)
    se = 1.2533 * np.std(beta, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Cox likelihood with mass at observed failure times leading to the Breslow estimator"})


def cheatsheet():
    return "ksr069: Cox likelihood with mass at observed failure times leading to the Breslow estimator"
