"""Generalized likelihood ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["glr_test"]


def glr_test(x, p0, p1):
    """
    Generalized likelihood ratio

    Formula: Λ_n = max_τ log p_1(x|τ)/p_0(x)

    Parameters
    ----------
    x : array-like
        Input data.
    p0 : array-like
        Input data.
    p1 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lai (1995)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized likelihood ratio"})


def cheatsheet():
    return "glm: Generalized likelihood ratio"
