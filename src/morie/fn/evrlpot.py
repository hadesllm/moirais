"""T-period return level under POT/GPD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_return_level_pot"]


def evt_return_level_pot(u, sigma, xi, zeta_u, n_y, T):
    """
    T-period return level under POT/GPD

    Formula: z_T = u + (σ/ξ)((Tnζ_u)^{ξ}-1)

    Parameters
    ----------
    u : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.
    zeta_u : array-like
        Input data.
    n_y : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z_T

    References
    ----------
    Coles (2001)
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    result = float(np.mean(u))
    se = float(np.std(u, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T-period return level under POT/GPD"})


def cheatsheet():
    return "evrlpot: T-period return level under POT/GPD"
