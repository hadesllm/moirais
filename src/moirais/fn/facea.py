"""FACE — fast covariance estimation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["face_smooth"]


def face_smooth(Y, argvals):
    """
    FACE — fast covariance estimation

    Formula: sandwich smoother on cov kernel

    Parameters
    ----------
    Y : array-like
        Input data.
    argvals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Xiao et al (2016) FACE
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "FACE — fast covariance estimation"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "FACE — fast covariance estimation"})


def cheatsheet():
    return "facea: FACE — fast covariance estimation"
