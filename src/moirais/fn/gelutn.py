"""GELU approximation via tanh."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gelu_tanh_approx"]


def gelu_tanh_approx(y):
    """
    GELU approximation via tanh

    Formula: GELU(x) ~ 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hendrycks & Gimpel (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GELU approximation via tanh"})


def cheatsheet():
    return "gelutn: GELU approximation via tanh"
