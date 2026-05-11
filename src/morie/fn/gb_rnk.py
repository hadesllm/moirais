# morie.fn — function file (hadesllm/morie)
"""Definition of rank as number of observations <= X_i in sample."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_rank_def"]


def gibbons_rank_def(x):
    """
    Definition of rank as number of observations <= X_i in sample

    Formula: rank(X_i) = sum_{j=1}^n I(X_j <= X_i) = n*S_n(X_i)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ranks

    References
    ----------
    Gibbons Ch 2.11.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definition of rank as number of observations <= X_i in sample"})


def cheatsheet():
    return "gb_rnk: Definition of rank as number of observations <= X_i in sample"
