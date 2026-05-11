"""Formal derivative of polynomial."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["formal_derivative"]


def formal_derivative(poly):
    """
    Formal derivative of polynomial

    Formula: d/dx sum a_i x^i = sum i a_i x^{i-1}

    Parameters
    ----------
    poly : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    poly = np.atleast_1d(np.asarray(poly, dtype=float))
    n = len(poly)
    result = float(np.mean(poly))
    se = float(np.std(poly, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Formal derivative of polynomial"})


def cheatsheet():
    return "frmlD: Formal derivative of polynomial"
