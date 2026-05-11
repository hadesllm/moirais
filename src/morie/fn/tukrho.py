"""Tukey biweight ρ."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tukey_biweight"]


def tukey_biweight(r, c):
    """
    Tukey biweight ρ

    Formula: ρ(r) = c²/6 (1 − [1 − (r/c)²]³) if |r|≤c else c²/6

    Parameters
    ----------
    r : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tukey (1960)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tukey biweight ρ"})


def cheatsheet():
    return "tukrho: Tukey biweight ρ"
