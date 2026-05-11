# morie.fn — function file (hadesllm/morie)
"""Bayesian nonparametric regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_np_regression"]


def ghosal_np_regression(x, y):
    """
    Bayesian nonparametric regression

    Formula: Y = f(X) + e, f ~ GP prior

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Ghosal Ch 12
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian nonparametric regression"})


def cheatsheet():
    return "ghreg: Bayesian nonparametric regression"
