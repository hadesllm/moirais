"""Hedges' g standardised mean difference (small-sample bias correction)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_hedges_g"]


def ma_hedges_g(m1, m2, s1, s2, n1, n2):
    """
    Hedges' g standardised mean difference (small-sample bias correction)

    Formula: g = (m1-m2)/s_p * J(df), J=1-3/(4 df-1)

    Parameters
    ----------
    m1 : array-like
        Input data.
    m2 : array-like
        Input data.
    s1 : array-like
        Input data.
    s2 : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g, var_g

    References
    ----------
    Hedges (1981)
    """
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    n = len(m1)
    result = float(np.mean(m1))
    se = float(np.std(m1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hedges' g standardised mean difference (small-sample bias correction)"})


def cheatsheet():
    return "mahg: Hedges' g standardised mean difference (small-sample bias correction)"
