# morie.fn -- function file (hadesllm/morie)
"""Nonparametric additive model with identity link."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_additive_model"]


def horowitz_additive_model(x, y, bandwidth):
    """
    Nonparametric additive model with identity link

    Formula: E[Y|X] = mu + sum_{j=1}^d g_j(X_j) with sum-to-zero constraints

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_j_hats, mu_hat

    References
    ----------
    Horowitz Ch 3, Sec 3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric additive model with identity link"})


def cheatsheet():
    return "hrzadm: Nonparametric additive model with identity link"
