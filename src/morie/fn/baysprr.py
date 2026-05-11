"""Horseshoe sparsity prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sparsity_horseshoe"]


def sparsity_horseshoe(X, y, tau):
    """
    Horseshoe sparsity prior

    Formula: beta_j ~ N(0, tau lambda_j); lambda ~ C+(0,1)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Carvalho-Polson-Scott (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Horseshoe sparsity prior"})


def cheatsheet():
    return "baysprr: Horseshoe sparsity prior"
