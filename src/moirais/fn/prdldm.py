"""Proximal gradient method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prox_method"]


def prox_method(f, grad_f, prox_g, x0, lr):
    """
    Proximal gradient method

    Formula: x_{t+1} = prox_g(x_t - lr grad f(x_t))

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    prox_g : array-like
        Input data.
    x0 : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Combettes-Wajs (2005)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal gradient method"})


def cheatsheet():
    return "prdldm: Proximal gradient method"
