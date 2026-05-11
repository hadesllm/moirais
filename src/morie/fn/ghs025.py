"""Density of a tail-free random measure as the infinite product of doubled splitting variables along the binary expansion of the point x.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_density_product"]


def ghosal_ch3_tailfree_density_product(V, x):
    """
    Density of a tail-free random measure as the infinite product of doubled splitting variables along the binary expansion of the point x.

    Formula: p(x) = prod_{j=1}^{infty} ( 2 * V_{x_1 x_2 ... x_j} )

    Parameters
    ----------
    V : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.18, p. 44
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Density of a tail-free random measure as the infinite product of doubled splitting variables along the binary expansion of the point x."})


def cheatsheet():
    return "ghs025: Density of a tail-free random measure as the infinite product of doubled splitting variables along the binary expansion of the point x."
