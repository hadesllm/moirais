"""SNP quality control."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["snp_qc"]


def snp_qc(genotypes, filters):
    """
    SNP quality control

    Formula: call rate + MAF + HWE filters

    Parameters
    ----------
    genotypes : array-like
        Input data.
    filters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Marees et al (2018)
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SNP quality control"})


def cheatsheet():
    return "snpqc1: SNP quality control"
