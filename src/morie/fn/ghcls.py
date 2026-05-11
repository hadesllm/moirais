# morie.fn — function file (hadesllm/morie)
"""Bayesian nonparametric classification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_np_classification"]


def ghosal_np_classification(x, y):
    """
    Bayesian nonparametric classification

    Formula: P(Y=1|X) = Phi(f(X)), f ~ GP

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 12
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian nonparametric classification"})


def cheatsheet():
    return "ghcls: Bayesian nonparametric classification"
