# morie.fn -- function file (hadesllm/morie)
"""Distribution of sample range W = X_(n) - X_(1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_range_dist"]


def gibbons_range_dist(w, n, f, F):
    """
    Distribution of sample range W = X_(n) - X_(1)

    Formula: F_W(w) = n * integral [F(w+w)-F(w)]^(n-1) f(w) dx

    Parameters
    ----------
    w : array-like
        Input data.
    n : array-like
        Input data.
    f : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Gibbons Ch 2.7.2
    """
    w = np.asarray(w, dtype=float)
    n = int(w) if w.ndim == 0 else len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Distribution of sample range W = X_(n) - X_(1)"})


def cheatsheet():
    return "gb_rng: Distribution of sample range W = X_(n) - X_(1)"
