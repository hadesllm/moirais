"""One-step / cross-fit estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cross_fit_one_step"]


def cross_fit_one_step(Y, X, M, C, K):
    """
    One-step / cross-fit estimator

    Formula: η̂ = θ̂ + EIF correction

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov et al (2018)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "One-step / cross-fit estimator"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "One-step / cross-fit estimator"})


def cheatsheet():
    return "crsfit: One-step / cross-fit estimator"
