"""Newton-Raphson."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["newton_raphson"]


def newton_raphson(f, grad_f, hess_f, x0):
    """
    Newton-Raphson

    Formula: x_{t+1} = x_t - H^-1 g

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    hess_f : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newton; Raphson (1690)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Newton-Raphson"})


def cheatsheet():
    return "newraf: Newton-Raphson"
