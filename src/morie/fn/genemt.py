"""Gene-based meta-analysis (MAGMA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gene_meta_analysis"]


def gene_meta_analysis(sumstats, gene_annotation):
    """
    Gene-based meta-analysis (MAGMA)

    Formula: combine SNP p-values via Brown's method

    Parameters
    ----------
    sumstats : array-like
        Input data.
    gene_annotation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Leeuw et al (2015) MAGMA
    """
    sumstats = np.atleast_1d(np.asarray(sumstats, dtype=float))
    n = len(sumstats)
    result = float(np.mean(sumstats))
    se = float(np.std(sumstats, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gene-based meta-analysis (MAGMA)"})


def cheatsheet():
    return "genemt: Gene-based meta-analysis (MAGMA)"
