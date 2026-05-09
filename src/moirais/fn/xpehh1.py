"""Cross-population EHH (XP-EHH)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["xpehh"]


def xpehh(haplotypes_p1, haplotypes_p2):
    """
    Cross-population EHH (XP-EHH)

    Formula: log(integrated EHH_pop1 / EHH_pop2)

    Parameters
    ----------
    haplotypes_p1 : array-like
        Input data.
    haplotypes_p2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sabeti et al (2007)
    """
    haplotypes_p1 = np.atleast_1d(np.asarray(haplotypes_p1, dtype=float))
    n = len(haplotypes_p1)
    result = float(np.mean(haplotypes_p1))
    se = float(np.std(haplotypes_p1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-population EHH (XP-EHH)"})


def cheatsheet():
    return "xpehh1: Cross-population EHH (XP-EHH)"
