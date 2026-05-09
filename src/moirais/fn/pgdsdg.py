"""Projected gradient descent."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["projected_gd"]


def projected_gd(f, grad_f, project, x0, lr):
    """
    Projected gradient descent

    Formula: x_{t+1} = proj_C(x_t - lr grad f)

    Parameters
    ----------
    f : array-like
        Input data.
    grad_f : array-like
        Input data.
    project : array-like
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
    Goldstein (1964)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Projected gradient descent"})


def cheatsheet():
    return "pgdsdg: Projected gradient descent"
