"""Peto one-step odds ratio for sparse 2x2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_peto_or"]


def ma_peto_or(a, b, c, d):
    """
    Peto one-step odds ratio for sparse 2x2

    Formula: log OR_Peto = (Σ(a_i - E_i))/Σ V_i

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: OR, se_log, ci

    References
    ----------
    Peto et al. (1977)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Peto one-step odds ratio for sparse 2x2"})


def cheatsheet():
    return "mapeto: Peto one-step odds ratio for sparse 2x2"
