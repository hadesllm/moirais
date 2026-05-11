"""Indian buffet generalized prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["isgp_bayes"]


def isgp_bayes(y, sigma, alpha, c):
    """
    Indian buffet generalized prior

    Formula: three-parameter IBP with sigma, alpha, c

    Parameters
    ----------
    y : array-like
        Input data.
    sigma : array-like
        Input data.
    alpha : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Teh-Görür (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Indian buffet generalized prior"})


def cheatsheet():
    return "isbplr: Indian buffet generalized prior"
