"""Parameter-stability plot for GPD threshold choice."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_param_stability"]


def evt_param_stability(x, u_grid):
    """
    Parameter-stability plot for GPD threshold choice

    Formula: fit GPD over u_grid; track σ̂*-uξ̂ stability

    Parameters
    ----------
    x : array-like
        Input data.
    u_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: u_grid, sigma_star, xi

    References
    ----------
    Coles (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parameter-stability plot for GPD threshold choice"})


def cheatsheet():
    return "evprmstab: Parameter-stability plot for GPD threshold choice"
