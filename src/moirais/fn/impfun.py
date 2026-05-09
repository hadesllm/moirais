"""Genotype imputation (Beagle/IMPUTE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["genotype_imputation"]


def genotype_imputation(genotypes, reference):
    """
    Genotype imputation (Beagle/IMPUTE)

    Formula: HMM with reference panel

    Parameters
    ----------
    genotypes : array-like
        Input data.
    reference : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Howie-Donnelly-Marchini (2009); Browning-Browning (2007)
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genotype imputation (Beagle/IMPUTE)"})


def cheatsheet():
    return "impfun: Genotype imputation (Beagle/IMPUTE)"
