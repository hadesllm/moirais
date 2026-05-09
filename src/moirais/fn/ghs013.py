"""Posterior Dirichlet distribution of the first k weights after marginalizing over the remaining cells, depending only on (N_1, ..., N_k).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_countable_dirichlet_posterior_k"]


def ghosal_ch3_countable_dirichlet_posterior_k(alpha_j, N_j, k, n):
    """
    Posterior Dirichlet distribution of the first k weights after marginalizing over the remaining cells, depending only on (N_1, ..., N_k).

    Formula: Dir( k+1; alpha_1 + N_1, ..., alpha_k + N_k, sum_{j=k+1}^{infty} alpha_j + n - sum_{j=1}^{k} N_j )

    Parameters
    ----------
    alpha_j : array-like
        Input data.
    N_j : array-like
        Input data.
    k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: posterior

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.6, p. 32
    """
    alpha_j = np.atleast_1d(np.asarray(alpha_j, dtype=float))
    n = len(alpha_j)
    result = float(np.mean(alpha_j))
    se = float(np.std(alpha_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior Dirichlet distribution of the first k weights after marginalizing over the remaining cells, depending only on (N_1, ..., N_k)."})


def cheatsheet():
    return "ghs013: Posterior Dirichlet distribution of the first k weights after marginalizing over the remaining cells, depending only on (N_1, ..., N_k)."
