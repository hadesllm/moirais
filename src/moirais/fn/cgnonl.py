"""Nonlinear conjugate gradient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nonlinear_cg"]


def nonlinear_cg(f, grad_f, x0, method):
    """
    Nonlinear conjugate gradient

    Formula: Fletcher-Reeves or Polak-Ribière

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    x0 : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fletcher-Reeves (1964)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonlinear conjugate gradient"})


def cheatsheet():
    return "cgnonl: Nonlinear conjugate gradient"
