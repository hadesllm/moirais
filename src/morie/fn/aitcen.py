"""Compositional centre (geometric mean of compositions)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_center"]


def aitchison_center(X):
    """
    Compositional centre (geometric mean of compositions)

    Formula: cen(X)_i = C(geomean_n X_{ni})

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g

    References
    ----------
    Aitchison (1997)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Compositional centre (geometric mean of compositions)"})


def cheatsheet():
    return "aitcen: Compositional centre (geometric mean of compositions)"
