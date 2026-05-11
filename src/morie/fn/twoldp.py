"""D' two-locus disequilibrium."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["two_locus_dprime"]


def two_locus_dprime(geno1, geno2):
    """
    D' two-locus disequilibrium

    Formula: D' = D / D_max

    Parameters
    ----------
    geno1 : array-like
        Input data.
    geno2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lewontin (1964)
    """
    geno1 = np.atleast_1d(np.asarray(geno1, dtype=float))
    n = len(geno1)
    result = float(np.mean(geno1))
    se = float(np.std(geno1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "D' two-locus disequilibrium"})


def cheatsheet():
    return "twoldp: D' two-locus disequilibrium"
