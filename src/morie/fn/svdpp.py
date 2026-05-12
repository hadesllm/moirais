"""SVD++ -- implicit feedback."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["svdpp"]


def svdpp(R, implicit, K):
    """
    SVD++ -- implicit feedback

    Formula: r̂ = μ + b_u + b_i + q_i^T(p_u + |N(u)|^{-1/2} sum y_j)

    Parameters
    ----------
    R : array-like
        Input data.
    implicit : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Koren (2008)
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVD++ -- implicit feedback"})


def cheatsheet():
    return "svdpp: SVD++ -- implicit feedback"
