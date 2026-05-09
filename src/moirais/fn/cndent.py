"""Conditional entropy H(Y|X)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["conditional_entropy"]


def conditional_entropy(pxy):
    """
    Conditional entropy H(Y|X)

    Formula: H(Y|X) = H(X,Y) - H(X)

    Parameters
    ----------
    pxy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cover-Thomas (2006)
    """
    pxy = np.atleast_1d(np.asarray(pxy, dtype=float))
    n = len(pxy)
    result = float(np.mean(pxy))
    se = float(np.std(pxy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional entropy H(Y|X)"})


def cheatsheet():
    return "cndent: Conditional entropy H(Y|X)"
