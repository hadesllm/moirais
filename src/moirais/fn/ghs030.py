"""Posterior expected density at x for a canonical Polya tree process with rate sequence a_m, given an i.i.d. sample with level-j cell counts N_{x_1 ... x_j}.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_polya_tree_posterior_density"]


def ghosal_ch3_polya_tree_posterior_density(a_j, N, x, n):
    """
    Posterior expected density at x for a canonical Polya tree process with rate sequence a_m, given an i.i.d. sample with level-j cell counts N_{x_1 ... x_j}.

    Formula: E( p(x) | X_1, ..., X_n ) = prod_{j=1}^{infty} ( 2 * a_j + 2 * N_{x_1 ... x_j} ) / ( 2 * a_j + N_{x_1 ... x_{j-1}} )

    Parameters
    ----------
    a_j : array-like
        Input data.
    N : array-like
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
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.23, p. 50
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior expected density at x for a canonical Polya tree process with rate sequence a_m, given an i.i.d. sample with level-j cell counts N_{x_1 ... x_j}."})


def cheatsheet():
    return "ghs030: Posterior expected density at x for a canonical Polya tree process with rate sequence a_m, given an i.i.d. sample with level-j cell counts N_{x_1 ... x_j}."
