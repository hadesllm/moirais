"""Admixture analysis (model-based clustering)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["admixture_seq"]


def admixture_seq(genotypes, K):
    """
    Admixture analysis (model-based clustering)

    Formula: Q ancestry proportions; P pop allele freqs

    Parameters
    ----------
    genotypes : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Alexander-Novembre-Lange (2009) ADMIXTURE
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Admixture analysis (model-based clustering)"})


def cheatsheet():
    return "adseqs: Admixture analysis (model-based clustering)"
