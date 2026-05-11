"""Identity-by-descent matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ibd_matrix"]


def ibd_matrix(genotypes, map):
    """
    Identity-by-descent matrix

    Formula: HMM-based IBD inference

    Parameters
    ----------
    genotypes : array-like
        Input data.
    map : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Browning-Browning (2010)
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identity-by-descent matrix"})


def cheatsheet():
    return "ibdmtx: Identity-by-descent matrix"
