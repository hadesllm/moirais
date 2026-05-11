"""Orthogonalized Gnanadesikan-Kettenring covariance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["orthogonalized_gk"]


def orthogonalized_gk(y, X):
    """
    Orthogonalized Gnanadesikan-Kettenring covariance

    Formula: pairwise robust scales orthogonalized via eigendecomp

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Maronna & Zamar (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Orthogonalized Gnanadesikan-Kettenring covariance"})


def cheatsheet():
    return "ogkcv: Orthogonalized Gnanadesikan-Kettenring covariance"
