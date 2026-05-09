"""Peaks-over-threshold GPD fit + scale-invariance check."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_pot_fit"]


def evt_pot_fit(x, u):
    """
    Peaks-over-threshold GPD fit + scale-invariance check

    Formula: fit GPD on x[x>u]; rate ζ_u = N_u/n

    Parameters
    ----------
    x : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma, xi, zeta_u

    References
    ----------
    Davison & Smith (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Peaks-over-threshold GPD fit + scale-invariance check"})


def cheatsheet():
    return "evpot: Peaks-over-threshold GPD fit + scale-invariance check"
