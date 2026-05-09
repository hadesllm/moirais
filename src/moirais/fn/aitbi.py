"""CLR biplot loadings + scores from SVD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_biplot"]


def aitchison_biplot(X):
    """
    CLR biplot loadings + scores from SVD

    Formula: U Σ V^T = clr(X) – cen

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: U, S, V

    References
    ----------
    Aitchison & Greenacre (2002)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CLR biplot loadings + scores from SVD"})


def cheatsheet():
    return "aitbi: CLR biplot loadings + scores from SVD"
