"""Quasi-Biennial Oscillation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["qbo"]


def qbo(U30):
    """
    Quasi-Biennial Oscillation

    Formula: 30 hPa zonal wind at equator

    Parameters
    ----------
    U30 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Reed et al (1961)
    """
    U30 = np.atleast_1d(np.asarray(U30, dtype=float))
    n = len(U30)
    result = float(np.mean(U30))
    se = float(np.std(U30, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quasi-Biennial Oscillation"})


def cheatsheet():
    return "qboIdx: Quasi-Biennial Oscillation"
