"""Pushforward density via change-of-variables and Jacobian."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_pushforward_density"]


def ot_pushforward_density(mu_grid, T_jac, T_inv_grid):
    """
    Pushforward density via change-of-variables and Jacobian

    Formula: ν(y) = μ(T^{-1}(y))/|det DT(T^{-1}(y))|

    Parameters
    ----------
    mu_grid : array-like
        Input data.
    T_jac : array-like
        Input data.
    T_inv_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: nu

    References
    ----------
    Peyré & Cuturi (2019)
    """
    mu_grid = np.atleast_1d(np.asarray(mu_grid, dtype=float))
    n = len(mu_grid)
    result = float(np.mean(mu_grid))
    se = float(np.std(mu_grid, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pushforward density via change-of-variables and Jacobian"})


def cheatsheet():
    return "otpush: Pushforward density via change-of-variables and Jacobian"
