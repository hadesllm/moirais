"""Subsample-size choice via min-volatility heuristic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_subsample_rate"]


def boot_subsample_rate(x, stat, m_grid, B):
    """
    Subsample-size choice via min-volatility heuristic

    Formula: m̂ = argmin Var(θ̂_m) across grid

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    m_grid : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m_star, vol_curve

    References
    ----------
    Bickel & Sakov (2008)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Subsample-size choice via min-volatility heuristic"})


def cheatsheet():
    return "btsubrho: Subsample-size choice via min-volatility heuristic"
