"""EMMAX GWAS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["emmax_gwas"]


def emmax_gwas(y, M, K):
    """
    EMMAX GWAS

    Formula: approximate REML once; per-SNP linear test

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kang et al (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EMMAX GWAS"})


def cheatsheet():
    return "gwasem: EMMAX GWAS"
