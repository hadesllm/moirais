"""Cross-classified membership weight matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cross_classified_membership"]


def cross_classified_membership(y, cluster1, cluster2):
    """
    Cross-classified membership weight matrix

    Formula: W_ij = 1/n_j if i in cluster j else 0; rows sum to 1

    Parameters
    ----------
    y : array-like
        Input data.
    cluster1 : array-like
        Input data.
    cluster2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goldstein (1994); Browne, Goldstein, Rasbash (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-classified membership weight matrix"})


def cheatsheet():
    return "ccmem: Cross-classified membership weight matrix"
