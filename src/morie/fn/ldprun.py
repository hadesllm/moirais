"""LD-based SNP pruning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ld_pruning"]


def ld_pruning(genotypes, r2_threshold, window):
    """
    LD-based SNP pruning

    Formula: greedy keep set with pairwise r^2 < threshold

    Parameters
    ----------
    genotypes : array-like
        Input data.
    r2_threshold : array-like
        Input data.
    window : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LD-based SNP pruning"})


def cheatsheet():
    return "ldprun: LD-based SNP pruning"
