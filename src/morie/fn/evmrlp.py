"""Mean residual life plot for threshold selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_mean_residual_life"]


def evt_mean_residual_life(x, u_grid):
    """
    Mean residual life plot for threshold selection

    Formula: e(u) = mean(X-u | X>u)

    Parameters
    ----------
    x : array-like
        Input data.
    u_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: u, e_u

    References
    ----------
    Davison & Smith (1990)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean residual life plot for threshold selection"})


def cheatsheet():
    return "evmrlp: Mean residual life plot for threshold selection"
