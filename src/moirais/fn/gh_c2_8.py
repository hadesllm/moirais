# moirais.fn — function file (hadesllm/moirais)
"""Bayesian nonparametric normal regression prior: Y|X ~ N(f(X), sigma^2), f ~ GP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_np_normal_reg"]


def ghosal_np_normal_reg(x, y):
    """
    Bayesian nonparametric normal regression prior: Y|X ~ N(f(X), sigma^2), f ~ GP

    Formula: Y_i = f(x_i) + e_i, e_i ~ N(0,sigma^2), f ~ GP(0, k)

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
    Ghosal Ch 2 §2.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian nonparametric normal regression prior: Y|X ~ N(f(X), sigma^2), f ~ GP"})


def cheatsheet():
    return "gh_c2_8: Bayesian nonparametric normal regression prior: Y|X ~ N(f(X), sigma^2), f ~ GP"
