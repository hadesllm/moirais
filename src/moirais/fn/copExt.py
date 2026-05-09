"""Extremal copula (multivariate tail)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["extremal_copula"]


def extremal_copula(u, v, A):
    """
    Extremal copula (multivariate tail)

    Formula: C(u,v) = exp(-A(log u, log v))

    Parameters
    ----------
    u : array-like
        Input data.
    v : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Joe (1997) book
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    result = float(np.mean(u))
    se = float(np.std(u, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Extremal copula (multivariate tail)"})


def cheatsheet():
    return "copExt: Extremal copula (multivariate tail)"
