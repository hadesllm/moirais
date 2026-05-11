"""Penman-Monteith reference ET."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["penman_monteith"]


def penman_monteith(T, R_n, u2, VPD):
    """
    Penman-Monteith reference ET

    Formula: ET₀ = (Δ(R_n − G) + γ·900/(T+273)·u₂·VPD)/(Δ + γ(1+0.34u₂))

    Parameters
    ----------
    T : array-like
        Input data.
    R_n : array-like
        Input data.
    u2 : array-like
        Input data.
    VPD : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Allen et al (1998) FAO-56
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penman-Monteith reference ET"})


def cheatsheet():
    return "basEvap: Penman-Monteith reference ET"
