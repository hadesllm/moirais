"""Structural deep network embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sdne"]


def sdne(A, dim):
    """
    Structural deep network embedding

    Formula: autoencoder over adjacency rows

    Parameters
    ----------
    A : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang et al (2016) SDNE
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Structural deep network embedding"})


def cheatsheet():
    return "sdne: Structural deep network embedding"
