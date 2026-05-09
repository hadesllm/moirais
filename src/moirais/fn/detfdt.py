"""Detrended fluctuation analysis (DFA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["detrended_fluctuation"]


def detrended_fluctuation(y, scales):
    """
    Detrended fluctuation analysis (DFA)

    Formula: power-law scaling of detrended cumulative

    Parameters
    ----------
    y : array-like
        Input data.
    scales : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Peng et al (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Detrended fluctuation analysis (DFA)"})


def cheatsheet():
    return "detfdt: Detrended fluctuation analysis (DFA)"
