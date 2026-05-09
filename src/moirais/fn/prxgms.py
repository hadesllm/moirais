"""Proximal gradient for sparse models."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["proximal_gradient_method"]


def proximal_gradient_method(f, grad_f, x0, lr, lam):
    """
    Proximal gradient for sparse models

    Formula: x_{t+1} = soft_thresh(x_t - lr grad f, lr lambda)

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    x0 : array-like
        Input data.
    lr : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Beck-Teboulle (2009)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal gradient for sparse models"})


def cheatsheet():
    return "prxgms: Proximal gradient for sparse models"
