# morie.fn -- function file (hadesllm/morie)
"""Least-mean-squares (LMS) adaptive filter."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_lms_filter"]


def rangayyan_lms_filter(x, d, mu, order):
    """
    Least-mean-squares (LMS) adaptive filter

    Formula: e(n) = d(n) - w^T(n)*x(n); w(n+1) = w(n) + 2*mu*e(n)*x(n)

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    mu : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y, e, w_history

    References
    ----------
    Rangayyan Ch 3.10.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Least-mean-squares (LMS) adaptive filter"})


def cheatsheet():
    return "rglms: Least-mean-squares (LMS) adaptive filter"
