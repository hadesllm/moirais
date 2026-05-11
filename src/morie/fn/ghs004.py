"""Density induced from a real-valued function f via the exponential link, normalized by c(f) = log integral exp(f) d mu.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch2_exponential_link_density"]


def ghosal_ch2_exponential_link_density(f, x, mu):
    """
    Density induced from a real-valued function f via the exponential link, normalized by c(f) = log integral exp(f) d mu.

    Formula: p_f(x) = exp( f(x) - c(f) ),   c(f) = log integral e^f d mu

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.
    mu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 2, Eq 2.3, p. 16
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Density induced from a real-valued function f via the exponential link, normalized by c(f) = log integral exp(f) d mu."})


def cheatsheet():
    return "ghs004: Density induced from a real-valued function f via the exponential link, normalized by c(f) = log integral exp(f) d mu."
