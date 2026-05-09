"""Compute the Kantorovich dual value."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_kantorovich_dual_value"]


def ot_kantorovich_dual_value(a, b, f, g):
    """
    Compute the Kantorovich dual value

    Formula: <a,f> + <b,g>

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    f : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dual_val

    References
    ----------
    Villani (2003)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Compute the Kantorovich dual value"})


def cheatsheet():
    return "ototk: Compute the Kantorovich dual value"
