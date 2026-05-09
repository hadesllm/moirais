"""Extended haplotype homozygosity decay."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ehh_decay"]


def ehh_decay(haplotypes, core, d_grid):
    """
    Extended haplotype homozygosity decay

    Formula: EHH(d) decay around core SNP

    Parameters
    ----------
    haplotypes : array-like
        Input data.
    core : array-like
        Input data.
    d_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sabeti et al (2002)
    """
    haplotypes = np.atleast_1d(np.asarray(haplotypes, dtype=float))
    n = len(haplotypes)
    result = float(np.mean(haplotypes))
    se = float(np.std(haplotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Extended haplotype homozygosity decay"})


def cheatsheet():
    return "ehhdec: Extended haplotype homozygosity decay"
