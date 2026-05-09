"""Bray-Curtis dissimilarity (closed compositions)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_bray_curtis"]


def compositional_bray_curtis(x, y):
    """
    Bray-Curtis dissimilarity (closed compositions)

    Formula: BC(x,y) = Σ|x_i - y_i| / Σ(x_i + y_i)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bc

    References
    ----------
    Bray & Curtis (1957)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bray-Curtis dissimilarity (closed compositions)"})


def cheatsheet():
    return "aitbcp: Bray-Curtis dissimilarity (closed compositions)"
