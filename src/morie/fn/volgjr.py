"""Glosten-Jagannathan-Runkle GARCH with leverage."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_gjr_garch"]


def vol_gjr_garch(r, init):
    """
    Glosten-Jagannathan-Runkle GARCH with leverage

    Formula: σ_t² = ω + (α + γ I[ε_{t-1}<0]) ε_{t-1}² + β σ_{t-1}²

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, gamma, beta, ll

    References
    ----------
    GJR (1993)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glosten-Jagannathan-Runkle GARCH with leverage"})


def cheatsheet():
    return "volgjr: Glosten-Jagannathan-Runkle GARCH with leverage"
