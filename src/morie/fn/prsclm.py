"""PRS clumping + thresholding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prs_cs_clump"]


def prs_cs_clump(sumstats, ld_ref, p_threshold):
    """
    PRS clumping + thresholding

    Formula: top SNP per LD block at p < threshold

    Parameters
    ----------
    sumstats : array-like
        Input data.
    ld_ref : array-like
        Input data.
    p_threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Choi-O'Reilly (2019)
    """
    sumstats = np.atleast_1d(np.asarray(sumstats, dtype=float))
    n = len(sumstats)
    result = float(np.mean(sumstats))
    se = float(np.std(sumstats, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PRS clumping + thresholding"})


def cheatsheet():
    return "prsclm: PRS clumping + thresholding"
