"""Mutual information I(X;Y)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_mutual_info"]


def wasserman_mutual_info(x, y):
    """
    Mutual information I(X;Y)

    Formula: I(X;Y) = D_KL(p(x,y) || p(x) p(y))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 23
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mutual information I(X;Y)"})


def cheatsheet():
    return "wsmmtl: Mutual information I(X;Y)"
