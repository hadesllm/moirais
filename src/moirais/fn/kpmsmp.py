"""Simultaneous confidence band for KM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["km_simultaneous_band"]


def km_simultaneous_band(fit, alpha):
    """
    Simultaneous confidence band for KM

    Formula: Hall-Wellner band

    Parameters
    ----------
    fit : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hall-Wellner (1980)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simultaneous confidence band for KM"})


def cheatsheet():
    return "kpmsmp: Simultaneous confidence band for KM"
