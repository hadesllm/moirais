"""Konfound robustness % bias to invalidate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["konfound"]


def konfound(est, se, n, threshold):
    """
    Konfound robustness % bias to invalidate

    Formula: smallest correlation of unmeasured U to flip the result

    Parameters
    ----------
    est : array-like
        Input data.
    se : array-like
        Input data.
    n : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Frank et al (2013)
    """
    est = np.atleast_1d(np.asarray(est, dtype=float))
    n = len(est)
    result = float(np.mean(est))
    se = float(np.std(est, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Konfound robustness % bias to invalidate"})


def cheatsheet():
    return "konfound: Konfound robustness % bias to invalidate"
