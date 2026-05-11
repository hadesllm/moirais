"""Posterior mean of the j-th weight in a countable Dirichlet process given n observations with cell counts N_j.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_dirichlet_posterior_mean"]


def ghosal_ch3_dirichlet_posterior_mean(alpha_j, N_j, j, n):
    """
    Posterior mean of the j-th weight in a countable Dirichlet process given n observations with cell counts N_j.

    Formula: E(p_j | X_1, ..., X_n) = (alpha_j + N_j) / ( sum_{l=1}^{infty} alpha_l + n )

    Parameters
    ----------
    alpha_j : array-like
        Input data.
    N_j : array-like
        Input data.
    j : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.7, p. 32
    """
    alpha_j = np.atleast_1d(np.asarray(alpha_j, dtype=float))
    n = len(alpha_j)
    result = float(np.mean(alpha_j))
    se = float(np.std(alpha_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior mean of the j-th weight in a countable Dirichlet process given n observations with cell counts N_j."})


def cheatsheet():
    return "ghs014: Posterior mean of the j-th weight in a countable Dirichlet process given n observations with cell counts N_j."
