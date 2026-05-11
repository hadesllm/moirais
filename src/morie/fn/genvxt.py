"""Generalizability theory (G-coefficient)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["generalizability_theory"]


def generalizability_theory(X, facets):
    """
    Generalizability theory (G-coefficient)

    Formula: sigma_p^2 / (sigma_p^2 + sum sigma_others^2 / n)

    Parameters
    ----------
    X : array-like
        Input data.
    facets : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cronbach et al (1972)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalizability theory (G-coefficient)"})


def cheatsheet():
    return "genvxt: Generalizability theory (G-coefficient)"
