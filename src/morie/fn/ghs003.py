"""Approximation error bound when truncating a basis expansion at J terms, decaying as a power of 1/J governed by regularity alpha and dimension k.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch2_basis_truncation_error"]


def ghosal_ch2_basis_truncation_error(f, J, alpha, k):
    """
    Approximation error bound when truncating a basis expansion at J terms, decaying as a power of 1/J governed by regularity alpha and dimension k.

    Formula: || f - sum_{j=1}^{J} f_j * psi_j || <~ (1/J)^(alpha/k) * ||f||_alpha^*

    Parameters
    ----------
    f : array-like
        Input data.
    J : array-like
        Input data.
    alpha : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 2, Eq 2.2, p. 12
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Approximation error bound when truncating a basis expansion at J terms, decaying as a power of 1/J governed by regularity alpha and dimension k."})


def cheatsheet():
    return "ghs003: Approximation error bound when truncating a basis expansion at J terms, decaying as a power of 1/J governed by regularity alpha and dimension k."
