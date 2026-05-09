"""Identity-by-state matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ibs_matrix"]


def ibs_matrix(genotypes):
    """
    Identity-by-state matrix

    Formula: per-pair shared allele count / total

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
    Purcell et al (2007) PLINK
    """
    genotypes = np.atleast_1d(np.asarray(genotypes, dtype=float))
    n = len(genotypes)
    result = float(np.mean(genotypes))
    se = float(np.std(genotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identity-by-state matrix"})


def cheatsheet():
    return "ibsmtx: Identity-by-state matrix"
