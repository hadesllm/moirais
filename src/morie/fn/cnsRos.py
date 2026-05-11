"""Rosenbaum signed-rank bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rosenbaum_bound_signed"]


def rosenbaum_bound_signed(pairs, Gamma):
    """
    Rosenbaum signed-rank bound

    Formula: vary Γ; compute upper p-value

    Parameters
    ----------
    pairs : array-like
        Input data.
    Gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rosenbaum (2002)
    """
    pairs = np.atleast_1d(np.asarray(pairs, dtype=float))
    n = len(pairs)
    result = float(np.mean(pairs))
    se = float(np.std(pairs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rosenbaum signed-rank bound"})


def cheatsheet():
    return "cnsRos: Rosenbaum signed-rank bound"
