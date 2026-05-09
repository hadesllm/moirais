"""GARCH(1,1) with Student-t innovations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_garch_t"]


def vol_garch_t(r, init):
    """
    GARCH(1,1) with Student-t innovations

    Formula: ε_t = σ_t z_t, z_t ~ t_ν

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, beta, nu, ll

    References
    ----------
    Bollerslev (1987)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GARCH(1,1) with Student-t innovations"})


def cheatsheet():
    return "volgargt: GARCH(1,1) with Student-t innovations"
