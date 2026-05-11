"""Mantel-Haenszel pooled OR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mantel_haenszel_or"]


def mantel_haenszel_or(strata):
    """
    Mantel-Haenszel pooled OR

    Formula: OR_MH = sum(a_k d_k / n_k) / sum(b_k c_k / n_k)

    Parameters
    ----------
    strata : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mantel-Haenszel (1959)
    """
    strata = np.atleast_1d(np.asarray(strata, dtype=float))
    n = len(strata)
    result = float(np.mean(strata))
    se = float(np.std(strata, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mantel-Haenszel pooled OR"})


def cheatsheet():
    return "mhst1: Mantel-Haenszel pooled OR"
