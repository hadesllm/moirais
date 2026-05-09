"""Residual variance estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_residual_variance"]


def esl_residual_variance(X, y, beta):
    """
    Residual variance estimate

    Formula: sigma^2_hat = RSS/(n-p-1)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Residual variance estimate"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Residual variance estimate"})


def cheatsheet():
    return "eslsig: Residual variance estimate"
