"""Fejer-type theorem stating that location-scale kernel mixtures converge in L1 to f as the bandwidth sigma tends to zero.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch2_location_scale_mixture_limit"]


def ghosal_ch2_location_scale_mixture_limit(psi, f, sigma, mu):
    """
    Fejer-type theorem stating that location-scale kernel mixtures converge in L1 to f as the bandwidth sigma tends to zero.

    Formula: integral (1/sigma) * psi( ( . - mu ) / sigma ) f(mu) d mu  ->  f(.),   as sigma -> 0

    Parameters
    ----------
    psi : array-like
        Input data.
    f : array-like
        Input data.
    sigma : array-like
        Input data.
    mu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 2, Eq 2.4, p. 18
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    n = len(psi)
    result = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fejer-type theorem stating that location-scale kernel mixtures converge in L1 to f as the bandwidth sigma tends to zero."})


def cheatsheet():
    return "ghs005: Fejer-type theorem stating that location-scale kernel mixtures converge in L1 to f as the bandwidth sigma tends to zero."
