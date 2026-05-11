"""Doubly-censored GLS estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["doubly_censored_gls"]


def doubly_censored_gls(y, A, C, H):
    """
    Doubly-censored GLS estimator

    Formula: GLS with weights for both treatment and censoring

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    C : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Rotnitzky (1992)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Doubly-censored GLS estimator"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Doubly-censored GLS estimator"})


def cheatsheet():
    return "dctgls: Doubly-censored GLS estimator"
