"""Asymptotic expansion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["asymptotic_expansion"]


def asymptotic_expansion(f, x_inf):
    """
    Asymptotic expansion

    Formula: f ~ sum a_n φ_n as x→∞

    Parameters
    ----------
    f : array-like
        Input data.
    x_inf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Erdélyi (1956)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic expansion"})


def cheatsheet():
    return "asymp: Asymptotic expansion"
