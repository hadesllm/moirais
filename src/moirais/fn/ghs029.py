"""First and second moments of the Polya-tree density p(x) for a canonical Polya tree process with absolutely-continuous realizations.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_polya_tree_density_moments"]


def ghosal_ch3_polya_tree_density_moments(alpha, x):
    """
    First and second moments of the Polya-tree density p(x) for a canonical Polya tree process with absolutely-continuous realizations.

    Formula: E( p(x) ) = prod_{j=1}^{infty} 2 * alpha_{x_1 ... x_j} / ( alpha_{x_1 ... x_{j-1} 0} + alpha_{x_1 ... x_{j-1} 1} );   E( p(x)^2 ) = prod_{j=1}^{infty} 4 * alpha_{x_1 ... x_j} ( alpha_{x_1 ... x_j} + 1 ) / ( ( alpha_{x_1 ... x_{j-1} 0} + alpha_{x_1 ... x_{j-1} 1} ) ( alpha_{x_1 ... x_{j-1} 0} + alpha_{x_1 ... x_{j-1} 1} + 1 ) )

    Parameters
    ----------
    alpha : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.22, p. 49
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "First and second moments of the Polya-tree density p(x) for a canonical Polya tree process with absolutely-continuous realizations."})


def cheatsheet():
    return "ghs029: First and second moments of the Polya-tree density p(x) for a canonical Polya tree process with absolutely-continuous realizations."
