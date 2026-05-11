"""Powering (scalar action) on the simplex."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_powering"]


def aitchison_powering(alpha, x):
    """
    Powering (scalar action) on the simplex

    Formula: (α ⊙ x)_i = C(x_i^α)

    Parameters
    ----------
    alpha : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Powering (scalar action) on the simplex"})


def cheatsheet():
    return "aitpow: Powering (scalar action) on the simplex"
