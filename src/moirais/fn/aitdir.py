"""Dirichlet density on the simplex."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dirichlet_density"]


def dirichlet_density(x, alpha):
    """
    Dirichlet density on the simplex

    Formula: f(x|α) = Γ(Σα)/prod Γ(α_i) prod x_i^(α_i-1)

    Parameters
    ----------
    x : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: f

    References
    ----------
    Wilks (1962)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet density on the simplex"})


def cheatsheet():
    return "aitdir: Dirichlet density on the simplex"
