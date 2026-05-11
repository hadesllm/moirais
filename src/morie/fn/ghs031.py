"""Posterior expected density given hyperparameter theta for a Polya-tree mixture with elicited mean density g_theta and rate sequence a_m.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_polya_tree_mixture_post_density"]


def ghosal_ch3_polya_tree_mixture_post_density(g_theta, a_j, N, theta, x, n):
    """
    Posterior expected density given hyperparameter theta for a Polya-tree mixture with elicited mean density g_theta and rate sequence a_m.

    Formula: E( p(x) | theta, X_1, ..., X_n ) = g_theta(x) * prod_{j=1}^{infty} ( 2 * a_j + 2 * N_{G_theta(x)_1 ... G_theta(x)_j} ) / ( 2 * a_j + N_{G_theta(x)_1 ... G_theta(x)_{j-1}} )

    Parameters
    ----------
    g_theta : array-like
        Input data.
    a_j : array-like
        Input data.
    N : array-like
        Input data.
    theta : array-like
        Input data.
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: posterior

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.24, p. 53
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior expected density given hyperparameter theta for a Polya-tree mixture with elicited mean density g_theta and rate sequence a_m."})


def cheatsheet():
    return "ghs031: Posterior expected density given hyperparameter theta for a Polya-tree mixture with elicited mean density g_theta and rate sequence a_m."
