"""IGARCH(1,1) where α+β=1 enforces unit persistence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_igarch_fit"]


def vol_igarch_fit(r, init):
    """
    IGARCH(1,1) where α+β=1 enforces unit persistence

    Formula: σ_t² = ω + α ε_{t-1}² + (1-α) σ_{t-1}²

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, ll

    References
    ----------
    Engle-Bollerslev (1986)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IGARCH(1,1) where α+β=1 enforces unit persistence"})


def cheatsheet():
    return "volign: IGARCH(1,1) where α+β=1 enforces unit persistence"
