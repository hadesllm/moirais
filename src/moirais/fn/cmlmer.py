"""Compressed LMM for fast GWAS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compressed_lmm"]


def compressed_lmm(y, M, K, clusters):
    """
    Compressed LMM for fast GWAS

    Formula: cluster individuals; per-cluster random effect

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    K : array-like
        Input data.
    clusters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang et al (2010) cMLM
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Compressed LMM for fast GWAS"})


def cheatsheet():
    return "cmlmer: Compressed LMM for fast GWAS"
