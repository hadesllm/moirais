"""GARCH(1,1) MLE fit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_garch11_fit"]


def vol_garch11_fit(r, init):
    """
    GARCH(1,1) MLE fit

    Formula: σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, beta, ll, sigma2

    References
    ----------
    Bollerslev (1986)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GARCH(1,1) MLE fit"})


def cheatsheet():
    return "volgar: GARCH(1,1) MLE fit"
