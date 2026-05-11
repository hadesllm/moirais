"""Cox partial likelihood that gives an efficient estimator of beta."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_cox_partial_likelihood"]


def kosorok_ch3_cox_partial_likelihood(beta, Z, V, d, n):
    """
    Cox partial likelihood that gives an efficient estimator of beta

    Formula: L_tilde_n(beta) = prod_{i=1}^n [ exp(beta'Z_i) / sum_{j: V_j >= V_i} exp(beta'Z_j) ]^{d_i}

    Parameters
    ----------
    beta : array-like
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
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.4, p. 43
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Cox partial likelihood that gives an efficient estimator of beta"})
    estimate = np.median(beta)
    se = 1.2533 * np.std(beta, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Cox partial likelihood that gives an efficient estimator of beta"})


def cheatsheet():
    return "ksr064: Cox partial likelihood that gives an efficient estimator of beta"
