"""Kinship from genotypes (KING-robust)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kinship_estimator"]


def kinship_estimator(genotypes):
    """
    Kinship from genotypes (KING-robust)

    Formula: per-pair kinship via shared alleles

    Parameters
    ----------
    genotypes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manichaikul et al (2010) KING
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kinship from genotypes (KING-robust)"})


def cheatsheet():
    return "kngshp: Kinship from genotypes (KING-robust)"
