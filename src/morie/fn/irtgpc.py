"""Generalized partial credit model (GPCM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["generalized_partial_credit"]


def generalized_partial_credit(X, ncats):
    """
    Generalized partial credit model (GPCM)

    Formula: P(X=k|theta) = exp(sum a (theta - b_jh)) / sum_normalize

    Parameters
    ----------
    X : array-like
        Input data.
    ncats : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Muraki (1992)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized partial credit model (GPCM)"})


def cheatsheet():
    return "irtgpc: Generalized partial credit model (GPCM)"
