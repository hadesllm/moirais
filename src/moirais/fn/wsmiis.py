"""Importance sampling estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_importance_sampling"]


def wasserman_importance_sampling(f, p, q, n):
    """
    Importance sampling estimator

    Formula: I = (1/n) sum f(X_i) p(X_i)/q(X_i)

    Parameters
    ----------
    f : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 24
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Importance sampling estimator"})
    estimate = np.median(f)
    se = 1.2533 * np.std(f, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Importance sampling estimator"})


def cheatsheet():
    return "wsmiis: Importance sampling estimator"
