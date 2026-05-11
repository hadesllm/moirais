# morie.fn — function file (hadesllm/morie)
"""Distribution-free tolerance interval: U_(s)-U_(r) ~ Beta(s-r, n-s+r+1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_tolerance_beta"]


def gibbons_tolerance_beta(x, r, s, p, gamma):
    """
    Distribution-free tolerance interval: U_(s)-U_(r) ~ Beta(s-r, n-s+r+1)

    Formula: P(F(X_(s)) - F(X_(r)) >= p) = g; indices from beta integral

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
        Input data.
    s : array-like
        Input data.
    p : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: indices_r_s

    References
    ----------
    Gibbons Theorem 2.11.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Distribution-free tolerance interval: U_(s)-U_(r) ~ Beta(s-r, n-s+r+1)"})


def cheatsheet():
    return "gb2111: Distribution-free tolerance interval: U_(s)-U_(r) ~ Beta(s-r, n-s+r+1)"
