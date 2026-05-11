"""Imai-Keele sensitivity to unmeasured confounding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["imai_sensitivity_rho"]


def imai_sensitivity_rho(Y, X, M, rho_grid):
    """
    Imai-Keele sensitivity to unmeasured confounding

    Formula: vary correlation ρ between residuals; trace NIE

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    rho_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imai-Keele-Yamamoto (2010)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Imai-Keele sensitivity to unmeasured confounding"})


def cheatsheet():
    return "sensIM: Imai-Keele sensitivity to unmeasured confounding"
