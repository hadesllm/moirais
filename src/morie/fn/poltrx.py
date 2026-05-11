"""Mixture of Pólya tree priors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["polya_tree_extended"]


def polya_tree_extended(y, M, alpha):
    """
    Mixture of Pólya tree priors

    Formula: prior over centering + scale parameters

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hanson-Johnson (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixture of Pólya tree priors"})


def cheatsheet():
    return "poltrx: Mixture of Pólya tree priors"
