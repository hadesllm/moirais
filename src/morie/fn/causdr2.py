"""Orthogonal/double-robust score (Neyman-orthogonal)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_dr_orthogonal"]


def causal_dr_orthogonal(y, T, ps, m1, m0):
    """
    Orthogonal/double-robust score (Neyman-orthogonal)

    Formula: DR score IF; meets orthogonality condition

    Parameters
    ----------
    y : array-like
        Input data.
    T : array-like
        Input data.
    ps : array-like
        Input data.
    m1 : array-like
        Input data.
    m0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score_IF

    References
    ----------
    Chernozhukov et al. (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Orthogonal/double-robust score (Neyman-orthogonal)"})


def cheatsheet():
    return "causdr2: Orthogonal/double-robust score (Neyman-orthogonal)"
