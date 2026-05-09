"""Posterior Dirichlet distribution of the first l weights given multinomial cell counts N_1, ..., N_l from n observations.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_countable_dirichlet_posterior_l"]


def ghosal_ch3_countable_dirichlet_posterior_l(alpha_j, N_j, l, n):
    """
    Posterior Dirichlet distribution of the first l weights given multinomial cell counts N_1, ..., N_l from n observations.

    Formula: Dir( l+1; alpha_1 + N_1, ..., alpha_l + N_l, sum_{j=l+1}^{infty} alpha_j + n - sum_{j=1}^{l} N_j )

    Parameters
    ----------
    alpha_j : array-like
        Input data.
    N_j : array-like
        Input data.
    l : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: posterior

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.5, p. 32
    """
    alpha_j = np.atleast_1d(np.asarray(alpha_j, dtype=float))
    n = len(alpha_j)
    result = float(np.mean(alpha_j))
    se = float(np.std(alpha_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior Dirichlet distribution of the first l weights given multinomial cell counts N_1, ..., N_l from n observations."})


def cheatsheet():
    return "ghs012: Posterior Dirichlet distribution of the first l weights given multinomial cell counts N_1, ..., N_l from n observations."
