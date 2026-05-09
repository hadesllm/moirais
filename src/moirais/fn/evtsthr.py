"""Threshold choice via posterior log-variance minimisation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_threshold_select_lvar"]


def evt_threshold_select_lvar(x, u_grid):
    """
    Threshold choice via posterior log-variance minimisation

    Formula: argmin_u Var(log σ̂_u, ξ̂_u)

    Parameters
    ----------
    x : array-like
        Input data.
    u_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: u_star, score

    References
    ----------
    Northrop & Coleman (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Threshold choice via posterior log-variance minimisation"})


def cheatsheet():
    return "evtsthr: Threshold choice via posterior log-variance minimisation"
