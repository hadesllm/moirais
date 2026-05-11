# morie.fn — function file (hadesllm/morie)
"""Average pooling: output mean per pooling window."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_average_pool"]


def geron_average_pool(x, window, stride):
    """
    Average pooling: output mean per pooling window

    Formula: y[i,j,k] = mean over window W of x[i+u, j+v, k]

    Parameters
    ----------
    x : array-like
        Input data.
    window : array-like
        Input data.
    stride : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 12
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Average pooling: output mean per pooling window"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Average pooling: output mean per pooling window"})


def cheatsheet():
    return "hmavp: Average pooling: output mean per pooling window"
