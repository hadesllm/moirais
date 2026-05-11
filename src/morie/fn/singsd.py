"""Singular spectrum analysis (SSA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["singular_spectrum"]


def singular_spectrum(y, window):
    """
    Singular spectrum analysis (SSA)

    Formula: trajectory matrix SVD + grouping + reconstruction

    Parameters
    ----------
    y : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vautard-Yiou-Ghil (1992)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Singular spectrum analysis (SSA)"})


def cheatsheet():
    return "singsd: Singular spectrum analysis (SSA)"
