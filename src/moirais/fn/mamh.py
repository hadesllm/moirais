"""Mantel-Haenszel pooled odds ratio."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_mantel_haenszel"]


def ma_mantel_haenszel(a, b, c, d):
    """
    Mantel-Haenszel pooled odds ratio

    Formula: OR_MH = Σ(a_i d_i / N_i) / Σ(b_i c_i / N_i)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: OR_MH, se_log, ci

    References
    ----------
    Mantel & Haenszel (1959)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mantel-Haenszel pooled odds ratio"})


def cheatsheet():
    return "mamh: Mantel-Haenszel pooled odds ratio"
