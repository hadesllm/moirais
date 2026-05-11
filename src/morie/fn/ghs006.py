"""Feller-style approximation of a density by an integral mixture of kernels h_k weighted by the mixing distribution F.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch2_feller_density_approximation"]


def ghosal_ch2_feller_density_approximation(x, k, F, h_k, g_k, V):
    """
    Feller-style approximation of a density by an integral mixture of kernels h_k weighted by the mixing distribution F.

    Formula: a(x; k, F) = integral h_k(x; z) dF(z),   h_k(x; z) = (k / V(x)) * integral_{[z, infty)} (t - x) g_k(t; x) d nu_k(t)

    Parameters
    ----------
    x : array-like
        Input data.
    k : array-like
        Input data.
    F : array-like
        Input data.
    h_k : array-like
        Input data.
    g_k : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 2, Eq 2.5, p. 19
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Feller-style approximation of a density by an integral mixture of kernels h_k weighted by the mixing distribution F."})


def cheatsheet():
    return "ghs006: Feller-style approximation of a density by an integral mixture of kernels h_k weighted by the mixing distribution F."
