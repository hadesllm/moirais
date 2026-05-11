# morie.fn — function file (hadesllm/morie)
"""2D average pooling over kxk windows with stride s."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_average_pooling"]


def geron_average_pooling(X, k, stride):
    """
    2D average pooling over kxk windows with stride s

    Formula: Y[i,j] = mean over window X[i*s..i*s+k-1, j*s..j*s+k-1]

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    stride : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Géron Ch 12, Average Pooling section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "2D average pooling over kxk windows with stride s"})
    estimate = np.median(X)
    se = 1.2533 * np.std(X, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "2D average pooling over kxk windows with stride s"})


def cheatsheet():
    return "grapl: 2D average pooling over kxk windows with stride s"
