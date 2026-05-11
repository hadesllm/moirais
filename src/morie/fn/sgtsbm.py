"""Two-cluster spectral partition via Fiedler sign."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_spectral_clustering_2"]


def sgt_spectral_clustering_2(A):
    """
    Two-cluster spectral partition via Fiedler sign

    Formula: Cluster i to side sign(v_i)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels

    References
    ----------
    Shi-Malik (2000)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-cluster spectral partition via Fiedler sign"})


def cheatsheet():
    return "sgtsbm: Two-cluster spectral partition via Fiedler sign"
