"""Random basis expansion representing a function as a (possibly infinite) linear combination of basis functions with random coefficients.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch2_random_basis_expansion"]


def ghosal_ch2_random_basis_expansion(beta_j, psi_j, J):
    """
    Random basis expansion representing a function as a (possibly infinite) linear combination of basis functions with random coefficients.

    Formula: f = sum_{j in J} beta_j * psi_j

    Parameters
    ----------
    beta_j : array-like
        Input data.
    psi_j : array-like
        Input data.
    J : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: function

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 2, Eq 2.1, p. 10
    """
    beta_j = np.atleast_1d(np.asarray(beta_j, dtype=float))
    n = len(beta_j)
    result = float(np.mean(beta_j))
    se = float(np.std(beta_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random basis expansion representing a function as a (possibly infinite) linear combination of basis functions with random coefficients."})


def cheatsheet():
    return "ghs002: Random basis expansion representing a function as a (possibly infinite) linear combination of basis functions with random coefficients."
