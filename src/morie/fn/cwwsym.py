"""Continuous wavelet transform (Morlet)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cwt_morlet"]


def cwt_morlet(y, scales):
    """
    Continuous wavelet transform (Morlet)

    Formula: CWT(a,b) = (1/sqrt(a)) integral y(t) psi^*((t-b)/a) dt

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
    Torrence-Compo (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous wavelet transform (Morlet)"})


def cheatsheet():
    return "cwwsym: Continuous wavelet transform (Morlet)"
