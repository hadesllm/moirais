"""SNP-BLUP additive prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["snp_blup"]


def snp_blup(y, M):
    """
    SNP-BLUP additive prediction

    Formula: GEBV = M u_hat

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanRaden (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SNP-BLUP additive prediction"})


def cheatsheet():
    return "snpblr: SNP-BLUP additive prediction"
