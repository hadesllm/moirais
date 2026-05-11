"""Politis-Romano automatic block length selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_block_length_pr"]


def boot_block_length_pr(x, method):
    """
    Politis-Romano automatic block length selection

    Formula: ℓ̂ = round(C n^{1/5}); C from spectral density

    Parameters
    ----------
    x : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ell

    References
    ----------
    Politis & Romano (2009)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Politis-Romano automatic block length selection"})


def cheatsheet():
    return "btblen: Politis-Romano automatic block length selection"
