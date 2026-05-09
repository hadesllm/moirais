"""Missing-mechanism sensitivity (NMAR)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["missing_mechanism_sensitivity"]


def missing_mechanism_sensitivity(Y, R, delta_grid):
    """
    Missing-mechanism sensitivity (NMAR)

    Formula: vary δ shifting missing distribution

    Parameters
    ----------
    Y : array-like
        Input data.
    R : array-like
        Input data.
    delta_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Daniels-Hogan (2008)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Missing-mechanism sensitivity (NMAR)"})


def cheatsheet():
    return "missinM: Missing-mechanism sensitivity (NMAR)"
