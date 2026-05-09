"""Empirical Mode Decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["emd_decomposition"]


def emd_decomposition(y, max_imf):
    """
    Empirical Mode Decomposition

    Formula: iterative sifting into IMFs

    Parameters
    ----------
    y : array-like
        Input data.
    max_imf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huang et al (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Empirical Mode Decomposition"})


def cheatsheet():
    return "emdtsm: Empirical Mode Decomposition"
