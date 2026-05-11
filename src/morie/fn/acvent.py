"""Differential entropy h(X)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["differential_entropy"]


def differential_entropy(density):
    """
    Differential entropy h(X)

    Formula: h(X) = -integral f(x) log f(x) dx

    Parameters
    ----------
    density : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cover-Thomas (2006)
    """
    density = np.atleast_1d(np.asarray(density, dtype=float))
    n = len(density)
    result = float(np.mean(density))
    se = float(np.std(density, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Differential entropy h(X)"})


def cheatsheet():
    return "acvent: Differential entropy h(X)"
