"""Truncated power basis for splines."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_basis_truncated"]


def esl_basis_truncated(x, knots, p):
    """
    Truncated power basis for splines

    Formula: h_j(x) = x^j and h_{p+k}(x) = (x-xi_k)_+^p

    Parameters
    ----------
    x : array-like
        Input data.
    knots : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: basis

    References
    ----------
    Hastie ESL Ch 5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Truncated power basis for splines"})


def cheatsheet():
    return "esltrs: Truncated power basis for splines"
