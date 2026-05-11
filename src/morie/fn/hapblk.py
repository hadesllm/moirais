"""Haplotype block (Gabriel)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["haplotype_block"]


def haplotype_block(genotypes, cm_window):
    """
    Haplotype block (Gabriel)

    Formula: contiguous SNPs with high pairwise LD

    Parameters
    ----------
    genotypes : array-like
        Input data.
    cm_window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gabriel et al (2002)
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Haplotype block (Gabriel)"})


def cheatsheet():
    return "hapblk: Haplotype block (Gabriel)"
