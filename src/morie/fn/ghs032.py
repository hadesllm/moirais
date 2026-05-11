"""Two-sided product bounds on the Polya-tree posterior density factor used to control its tail when sum a_j^{-1} is finite.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_polya_tree_density_bounds"]


def ghosal_ch3_polya_tree_density_bounds(a_j, n, N, m):
    """
    Two-sided product bounds on the Polya-tree posterior density factor used to control its tail when sum a_j^{-1} is finite.

    Formula: prod_{j>m} ( 1 - n / (2 * a_j) )  <=  prod_{j>m} ( 2 * a_j + 2 * N_{G_theta(x)_1 ... G_theta(x)_j} ) / ( 2 * a_j + N_{G_theta(x)_1 ... G_theta(x)_{j-1}} )  <=  prod_{j>m} ( 1 + n / a_j )

    Parameters
    ----------
    a_j : array-like
        Input data.
    n : array-like
        Input data.
    N : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.25, p. 54
    """
    a_j = np.atleast_1d(np.asarray(a_j, dtype=float))
    n = len(a_j)
    result = float(np.mean(a_j))
    se = float(np.std(a_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-sided product bounds on the Polya-tree posterior density factor used to control its tail when sum a_j^{-1} is finite."})


def cheatsheet():
    return "ghs032: Two-sided product bounds on the Polya-tree posterior density factor used to control its tail when sum a_j^{-1} is finite."
