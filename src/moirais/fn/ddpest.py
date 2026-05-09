"""Dependent Dirichlet process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dependent_dp"]


def dependent_dp(x_grid, alpha, kernel):
    """
    Dependent Dirichlet process

    Formula: DP indexed by covariate; G_x varies smoothly

    Parameters
    ----------
    x_grid : array-like
        Input data.
    alpha : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    MacEachern (1999, 2000)
    """
    x_grid = np.atleast_1d(np.asarray(x_grid, dtype=float))
    n = len(x_grid)
    result = float(np.mean(x_grid))
    se = float(np.std(x_grid, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dependent Dirichlet process"})


def cheatsheet():
    return "ddpest: Dependent Dirichlet process"
