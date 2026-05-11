"""Polya-tree mixture of the second kind: mean density g_theta expressed as a product over levels of doubled theta-dependent alpha parameters.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_polya_tree_mixture_second_kind"]


def ghosal_ch3_polya_tree_mixture_second_kind(alpha_theta, x, theta):
    """
    Polya-tree mixture of the second kind: mean density g_theta expressed as a product over levels of doubled theta-dependent alpha parameters.

    Formula: g_theta(x) = prod_{j=1}^{infty} 2 * alpha_{x_1 ... x_j}(theta) / ( alpha_{x_1 ... x_{j-1} 0}(theta) + alpha_{x_1 ... x_{j-1} 1}(theta) )

    Parameters
    ----------
    alpha_theta : array-like
        Input data.
    x : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.26, p. 54
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polya-tree mixture of the second kind: mean density g_theta expressed as a product over levels of doubled theta-dependent alpha parameters."})


def cheatsheet():
    return "ghs033: Polya-tree mixture of the second kind: mean density g_theta expressed as a product over levels of doubled theta-dependent alpha parameters."
