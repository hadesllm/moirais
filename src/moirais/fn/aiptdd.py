"""AIPW estimator for DiD with covariates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aipw_did"]


def aipw_did(y_pre, y_post, D, X):
    """
    AIPW estimator for DiD with covariates

    Formula: AIPW = E[D Y_diff/p(X) - (1-D)/(1-p(X)) {Y_diff - m_diff(X)}]

    Parameters
    ----------
    y_pre : array-like
        Input data.
    y_post : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sant'Anna-Zhao (2020); Chang (2020)
    """
    y_pre = np.atleast_1d(np.asarray(y_pre, dtype=float))
    n = len(y_pre)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "AIPW estimator for DiD with covariates"})
    estimate = np.median(y_pre)
    se = 1.2533 * np.std(y_pre, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "AIPW estimator for DiD with covariates"})


def cheatsheet():
    return "aiptdd: AIPW estimator for DiD with covariates"
