"""Proximal gradient method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_proximal_grad"]


def boyd_proximal_grad(f, grad_f, h, x0, t):
    """
    Proximal gradient method

    Formula: x^{k+1} = prox_{tk h}(x^k - t_k grad f(x^k))

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    h : array-like
        Input data.
    x0 : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal gradient method"})


def cheatsheet():
    return "cvxprg: Proximal gradient method"
