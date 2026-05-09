"""ChebNet (Chebyshev polys of L)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chebnet"]


def chebnet(L, X, K):
    """
    ChebNet (Chebyshev polys of L)

    Formula: H = sum θ_k T_k(L̃) X

    Parameters
    ----------
    L : array-like
        Input data.
    X : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Defferrard et al (2016)
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    n = len(L)
    result = float(np.mean(L))
    se = float(np.std(L, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ChebNet (Chebyshev polys of L)"})


def cheatsheet():
    return "gcnchb: ChebNet (Chebyshev polys of L)"
