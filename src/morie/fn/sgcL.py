"""Simplified GCN (no nonlinearity)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgc"]


def sgc(A, X, K):
    """
    Simplified GCN (no nonlinearity)

    Formula: Y = softmax(Â^K X W)

    Parameters
    ----------
    A : array-like
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
    Wu et al (2019) SGC
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simplified GCN (no nonlinearity)"})


def cheatsheet():
    return "sgcL: Simplified GCN (no nonlinearity)"
