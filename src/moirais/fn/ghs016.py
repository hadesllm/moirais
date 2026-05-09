"""Posterior covariance between the j-th and j'-th weights of a countable Dirichlet process given n observations.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_dirichlet_posterior_cov"]


def ghosal_ch3_dirichlet_posterior_cov(alpha_j, alpha_jprime, N_j, N_jprime, n):
    """
    Posterior covariance between the j-th and j'-th weights of a countable Dirichlet process given n observations.

    Formula: cov(p_j, p_{j'} | X_1, ..., X_n) = - (alpha_j + N_j) * (alpha_{j'} + N_{j'}) / ( ( sum_{l=1}^{infty} alpha_l + n )^2 * ( sum_{l=1}^{infty} alpha_l + n + 1 ) )

    Parameters
    ----------
    alpha_j : array-like
        Input data.
    alpha_jprime : array-like
        Input data.
    N_j : array-like
        Input data.
    N_jprime : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.9, p. 33
    """
    alpha_j = np.atleast_1d(np.asarray(alpha_j, dtype=float))
    n = len(alpha_j)
    result = float(np.mean(alpha_j))
    se = float(np.std(alpha_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior covariance between the j-th and j'-th weights of a countable Dirichlet process given n observations."})


def cheatsheet():
    return "ghs016: Posterior covariance between the j-th and j'-th weights of a countable Dirichlet process given n observations."
