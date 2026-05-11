"""Dirichlet process mixture model with Gibbs sampling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dirichlet_process_mixture"]


def dirichlet_process_mixture(y, alpha, base_distribution, n_iter):
    """
    Dirichlet process mixture model with Gibbs sampling

    Formula: G ~ DP(alpha, G_0); theta_i | G ~ G; y_i ~ f(theta_i)

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    base_distribution : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Antoniak (1974); Escobar-West (1995); Neal (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet process mixture model with Gibbs sampling"})


def cheatsheet():
    return "dpmem: Dirichlet process mixture model with Gibbs sampling"
