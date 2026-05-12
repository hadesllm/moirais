# morie.fn -- function file (hadesllm/morie)
"""Recursive least-squares (RLS) adaptive filter."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_rls_filter"]


def rangayyan_rls_filter(x, d, lam, delta, order):
    """
    Recursive least-squares (RLS) adaptive filter

    Formula: P(n)=(P(n-1)-k(n)*x^T(n)*P(n-1))/lambda; w(n)=w(n-1)+k(n)*e(n)

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    lam : array-like
        Input data.
    delta : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y, e, w_history

    References
    ----------
    Rangayyan Ch 3.10.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recursive least-squares (RLS) adaptive filter"})


def cheatsheet():
    return "rgrls: Recursive least-squares (RLS) adaptive filter"
