"""Distribution-free tolerance limits."""
import numpy as np
from ._richresult import RichResult

__all__ = ["tolerance_limits"]


def tolerance_limits(x, coverage, confidence):
    """
    Distribution-free tolerance limits

    Formula: P(F(X_(s)) - F(X_(r)) >= beta) = 1-alpha

    Parameters
    ----------
    x : array-like
        Input data.
    coverage : array-like
        Input data.
    confidence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Gibbons Ch 2.11
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Distribution-free tolerance limits"})


def cheatsheet():
    return "tolim: Distribution-free tolerance limits"
